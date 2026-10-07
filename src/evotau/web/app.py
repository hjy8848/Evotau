"""Local, server-rendered Experiment Console backed by immutable EvoTau artifacts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .artifact_reader import ArtifactReader, ArtifactReadError
from .event_stream import EventJournal, EventJournalError
from .run_manager import RunManager, RunManagerError
from .run_progress import read_run_progress
from .view_models import (
    budget_view,
    current_strategy_for_episode,
    customer_strategy_view,
    db_state_trace_view,
    generation_view,
    service_strategy_view,
)

_WEB_ROOT = Path(__file__).resolve().parent


def create_app(
    *,
    project_root: str | Path | None = None,
    runs_root: str | Path | None = None,
    tau2_data_dir: str | Path | None = None,
    manager: RunManager | None = None,
) -> FastAPI:
    root = Path(project_root or _WEB_ROOT.parents[2]).expanduser().resolve()
    output_root = (
        Path(runs_root or root / "experiments" / "runs").expanduser().resolve()
    )
    reader = ArtifactReader(output_root, project_root=root)
    run_manager = manager or RunManager(root, output_root, tau2_data_dir=tau2_data_dir)
    journal = EventJournal()
    templates = Jinja2Templates(directory=str(_WEB_ROOT / "templates"))
    app = FastAPI(title="EvoTau Experiment Console", docs_url=None, redoc_url=None)
    app.mount(
        "/static", StaticFiles(directory=str(_WEB_ROOT / "static")), name="static"
    )
    app.state.csrf_tokens = set()
    app.state.csrf_lock = asyncio.Lock()

    @app.exception_handler(ArtifactReadError)
    async def hidden_bad_artifact(request: Request, _exc: ArtifactReadError):
        return page(request, "error.html", {
            "title": "Artifact 已隐藏",
            "message": "本地 artifact 缺失、损坏或校验不匹配，Console 已 fail closed。",
        }, status_code=422)

    @app.middleware("http")
    async def local_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    app.state.reader = reader
    app.state.run_manager = run_manager
    app.state.event_journal = journal

    def page(request: Request, template: str, context: dict[str, Any] | None = None, *, status_code: int = 200):
        payload = dict(context or {})
        current_run = payload.get("run")
        if isinstance(current_run, dict):
            payload.setdefault("command_index", _command_index(current_run))
            if current_run.get("status") == "running":
                payload.setdefault("csrf_token", issue_csrf_token())
        else:
            payload.setdefault("command_index", [])
        return templates.TemplateResponse(
            request=request,
            name=template,
            context={"request": request, **payload},
            status_code=status_code,
        )

    def load_run(run_id: str) -> dict[str, Any]:
        run = reader.get_run(run_id, live_status=run_manager.status_for_run(run_id))
        try:
            run["events"] = list(journal.sync(run))
        except (EventJournalError, OSError):
            run["events"] = []
            run["event_warning"] = "Console 事件日志损坏或不可写；Core 研究 artifact 不受影响。"
        run["budget"] = budget_view(run["budget"])
        run["generation_views"] = [
            generation_view(item) for item in run["generation_commits"]
        ]
        run["customer_strategy_views"] = {
            key: customer_strategy_view(value)
            for key, value in run["strategies"]["customer"].items()
        }
        run["service_strategy_views"] = {
            key: service_strategy_view(value)
            for key, value in run["strategies"]["service"].items()
        }
        run["current_customer_view"] = customer_strategy_view(
            run.get("current_customer_strategy")
        )
        run["current_service_view"] = service_strategy_view(
            run.get("current_service_strategy")
        )
        protocol_mode = run["manifest"].get(
            "enforce_communication_protocol",
            run["manifest"].get("communication_enforcement"),
        )
        run["communication_mode_label"] = (
            "Strict diagnostic" if protocol_mode is True else
            "Upstream default" if protocol_mode is False else "Unknown (legacy artifact)"
        )
        from .evolution_view import evolution_view

        run["evolution"] = evolution_view(reader, run)
        run["health"] = _health_view(run)
        run["human_events"] = _human_events(run)
        return run

    def issue_csrf_token() -> str:
        token = secrets.token_urlsafe(32)
        app.state.csrf_tokens.add(token)
        return token

    async def consume_csrf_token(token: str) -> bool:
        async with app.state.csrf_lock:
            for known in tuple(app.state.csrf_tokens):
                if hmac.compare_digest(known, token):
                    app.state.csrf_tokens.remove(known)
                    return True
        return False

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        try:
            runs = list(reader.list_runs())
        except ArtifactReadError:
            return page(request, "error.html", {
                "title": "结果目录不可读",
                "message": "实验目录中发现损坏或不安全的 artifact。请检查本地目录后刷新。",
            }, status_code=503)
        for run in runs:
            run["live_status"] = run_manager.status_for_run(run["run_id"])
            if run["live_status"]:
                run["status"] = run["live_status"]
        return page(request, "index.html", {
            "runs": runs,
            "tau2_data_dir": str(tau2_data_dir or os.environ.get("TAU2_DATA_DIR", "")),
            "config_phase0": "configs/mvp.yaml",
            "config_alternating": "configs/alternating-evolution.yaml",
            "csrf_token": issue_csrf_token(),
            "command_index": _command_index_for_summaries(runs),
        })

    @app.post("/preview", response_class=HTMLResponse)
    async def preview_run(
        request: Request,
    ):
        form = await _read_form(request)
        phase = form.get("phase", "")
        config_path = form.get("config_path", "")
        tau2_data_dir = form.get("tau2_data_dir", "")
        csrf_token = form.get("csrf_token", "")
        if not await consume_csrf_token(csrf_token):
            raise HTTPException(status_code=403, detail="form token expired; reload the dashboard")
        try:
            preview = run_manager.preview(
                phase=phase,
                config_path=config_path,
                tau2_data_dir=tau2_data_dir or None,
            )
        except RunManagerError as exc:
            return page(request, "error.html", {
                "title": "无法生成安全预览", "message": str(exc),
            }, status_code=400)
        return page(request, "preview.html", {
            "preview": preview.to_view(),
            "config_path": config_path,
            "tau2_data_dir": tau2_data_dir,
            "csrf_token": issue_csrf_token(),
        })

    @app.post("/start", response_class=HTMLResponse)
    async def start_run(
        request: Request,
    ):
        form = await _read_form(request)
        phase = form.get("phase", "")
        config_path = form.get("config_path", "")
        confirmed_manifest_sha256 = form.get("confirmed_manifest_sha256", "")
        tau2_data_dir = form.get("tau2_data_dir", "")
        confirm_experiment_id = form.get("confirm_experiment_id", "")
        csrf_token = form.get("csrf_token", "")
        if not await consume_csrf_token(csrf_token):
            raise HTTPException(status_code=403, detail="form token expired; reload the preview")
        try:
            preview = run_manager.preview(
                phase=phase, config_path=config_path,
                tau2_data_dir=tau2_data_dir or None,
            )
            if not hmac.compare_digest(confirm_experiment_id, preview.experiment_id):
                raise RunManagerError("请准确输入预览中的 Experiment ID 后再启动。")
            started = run_manager.start(
                phase=phase,
                config_path=config_path,
                confirmed_manifest_sha256=confirmed_manifest_sha256,
                confirmed_launch_sha256=form.get("confirmed_launch_sha256", ""),
                tau2_data_dir=tau2_data_dir or None,
            )
        except RunManagerError as exc:
            return page(request, "error.html", {
                "title": "Run 未启动", "message": str(exc),
            }, status_code=409)
        return RedirectResponse(f"/runs/{quote(started['run_id'], safe='')}", status_code=303)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_page(request: Request, run_id: str):
        try:
            run = load_run(run_id)
        except FileNotFoundError:
            return page(request, "error.html", {
                "title": "Run 尚不可用", "message": "没有找到对应的 manifest 或结果 artifact。",
            }, status_code=404)
        except ArtifactReadError:
            return page(request, "error.html", {
                "title": "Run 已隐藏", "message": "artifact 缺失、损坏或哈希不匹配，因此 Console 没有展示其内容。",
            }, status_code=422)
        pause_available = run["status"] == "running" and not run["phase"].startswith("0-")
        run["progress"] = read_run_progress(reader, run)
        return page(request, "run.html", {
            "run": run, "pause_available": pause_available,
            "csrf_token": issue_csrf_token(),
        })

    @app.get("/runs/{run_id}/evolution")
    async def evolution_page(request: Request, run_id: str):
        return page(request, "evolution.html", {"run": load_run(run_id)})

    @app.get("/runs/{run_id}/progress")
    async def run_progress(run_id: str):
        try:
            run = reader.get_run(run_id, live_status=run_manager.status_for_run(run_id))
            progress = read_run_progress(reader, run)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="run progress unavailable") from None
        if progress is None:
            raise HTTPException(status_code=404, detail="no alternating progress for this run")
        return JSONResponse(progress, headers={"Cache-Control": "no-store"})

    @app.post("/runs/{run_id}/pause")
    async def pause_run(run_id: str, request: Request):
        form = await _read_form(request)
        csrf_token = form.get("csrf_token", "")
        if not await consume_csrf_token(csrf_token):
            raise HTTPException(status_code=403, detail="form token expired; reload the run page")
        try:
            run_manager.pause(run_id)
        except RunManagerError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse(f"/runs/{quote(run_id, safe='')}", status_code=303)

    @app.get("/runs/{run_id}/events", response_class=HTMLResponse)
    async def events_stream(request: Request, run_id: str):
        try:
            run = load_run(run_id)
        except (FileNotFoundError, ArtifactReadError) as exc:
            raise HTTPException(status_code=404, detail="run events unavailable") from exc

        async def stream():
            sent: set[str] = set()
            for event in run["events"]:
                sent.add(event["event_id"])
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            while not await request.is_disconnected():
                await asyncio.sleep(1)
                try:
                    current_run = load_run(run_id)
                except (FileNotFoundError, ArtifactReadError):
                    yield ": artifact unavailable\n\n"
                    continue
                for event in current_run["events"]:
                    if event["event_id"] not in sent:
                        sent.add(event["event_id"])
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if not current_run["events"]:
                    yield ": keepalive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
        })

    @app.get("/runs/{run_id}/events.jsonl")
    async def events_jsonl(run_id: str):
        try:
            run = load_run(run_id)
        except (FileNotFoundError, ArtifactReadError) as exc:
            raise HTTPException(status_code=404, detail="run events unavailable") from exc
        return JSONResponse(run["events"])

    @app.get("/runs/{run_id}/episodes", response_class=HTMLResponse)
    async def episode_explorer(
        request: Request, run_id: str, generation: str = "", panel: str = "",
        customer: str = "", service: str = "", task: str = "", seed: str = "",
    ):
        try:
            run = load_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="run unavailable") from None
        episode_rows = list(run["episodes"])
        supplied = {
            "generation": generation, "panel": panel, "customer": customer,
            "service": service, "task": task, "seed": seed,
        }
        def matches(item: dict[str, Any]) -> bool:
            for key, value in supplied.items():
                if not value:
                    continue
                if key == "generation" and value not in {str(g) for g in item.get("generations", [item.get("generation")])}:
                    return False
                if key == "panel" and value not in item.get("panel_names", [item.get("panel_name") or ""]):
                    return False
                if key == "customer" and str(item.get("customer_strategy_id") or "") != value:
                    return False
                if key == "service" and str(item.get("service_strategy_id") or "") != value:
                    return False
                if key == "task" and str(item.get("task_id") or "") != value:
                    return False
                if key == "seed" and str(item.get("seed")) != value:
                    return False
            return True
        filtered = [item for item in episode_rows if matches(item)]
        options = {
            "generation": sorted({str(g) for item in episode_rows for g in item.get("generations", [item.get("generation")]) if g is not None}),
            "panel": sorted({p for item in episode_rows for p in item.get("panel_names", [item.get("panel_name")]) if p}),
            "customer": sorted({str(item["customer_strategy_id"]) for item in episode_rows if item.get("customer_strategy_id")}),
            "service": sorted({str(item["service_strategy_id"]) for item in episode_rows if item.get("service_strategy_id")}),
            "task": sorted({str(item["task_id"]) for item in episode_rows if item.get("task_id")}),
            "seed": sorted({str(item["seed"]) for item in episode_rows if item.get("seed") is not None}),
        }
        return page(request, "episodes.html", {
            "run": run, "episodes": filtered, "filters": supplied, "filter_options": options,
        })

    @app.get("/runs/{run_id}/episodes/{episode_id}", response_class=HTMLResponse)
    async def episode_page(request: Request, run_id: str, episode_id: str):
        try:
            item = reader.episode(run_id, episode_id)
        except FileNotFoundError:
            return page(request, "error.html", {
                "title": "Episode 不可用", "message": "该 episode 缺失，或仍处于封存的 Heldout panel。",
            }, status_code=404)
        except ArtifactReadError:
            return page(request, "error.html", {
                "title": "Episode 已隐藏", "message": "trajectory 与记录不一致，Console 已停止展示。",
            }, status_code=422)
        item["run"] = load_run(run_id)
        customer, service = current_strategy_for_episode(item["run"], item)
        return page(request, "episode.html", {
            "run": item["run"], "episode": item,
            "db_trace_view": db_state_trace_view(item.get("db_state_trace")),
            "customer_strategy": customer_strategy_view(customer),
            "service_strategy": service_strategy_view(service),
        })

    @app.get("/runs/{run_id}/strategies/{strategy_id}", response_class=HTMLResponse)
    async def strategy_page(request: Request, run_id: str, strategy_id: str, side: str = "customer"):
        if side not in {"customer", "service"}:
            raise HTTPException(status_code=400, detail="unknown strategy side")
        try:
            run = load_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="strategy unavailable") from None
        strategy = run["strategies"][side].get(strategy_id)
        if strategy is None:
            raise HTTPException(status_code=404, detail="strategy unavailable")
        view = customer_strategy_view(strategy) if side == "customer" else service_strategy_view(strategy)
        return page(request, "strategy.html", {
            "run": run, "strategy_id": strategy_id, "side": side,
            "strategy": strategy, "view": view,
        })

    @app.get("/compare", response_class=HTMLResponse)
    async def compare_runs(request: Request, run_a: str = "", run_b: str = ""):
        try:
            summaries = list(reader.list_runs())
        except ArtifactReadError:
            summaries = []
        runs = []
        selected = []
        for summary in summaries:
            summary["live_status"] = run_manager.status_for_run(summary["run_id"])
            runs.append(summary)
        if run_a and run_b:
            if run_a == run_b:
                raise HTTPException(status_code=400, detail="Choose two different runs")
            try:
                selected = [load_run(run_a), load_run(run_b)]
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="run unavailable") from None
        comparison = _compare_runs(*selected) if len(selected) == 2 else None
        return page(request, "compare.html", {
            "runs": runs, "selected": selected, "comparison": comparison,
            "command_index": _command_index_for_summaries(runs),
        })

    @app.get("/runs/{run_id}/artifacts", response_class=HTMLResponse)
    async def artifact_page(request: Request, run_id: str):
        try:
            run = load_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="run unavailable") from None
        return page(request, "artifacts.html", {"run": run, "artifacts": _artifact_inventory(run)})

    @app.get("/runs/{run_id}/heldout", response_class=HTMLResponse)
    async def heldout_page(request: Request, run_id: str):
        try:
            run = load_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="run unavailable") from None
        return page(request, "heldout.html", {"run": run})

    return app


def _command_index(run: dict[str, Any]) -> list[dict[str, str]]:
    run_id = quote(str(run["run_id"]), safe="")
    items = [{"title": str(run["experiment_id"]), "kind": "Run", "url": f"/runs/{run_id}",
              "summary": str(run.get("phase", ""))}]
    for episode in run.get("episodes", ()):
        episode_id = quote(str(episode["episode_id"]), safe="")
        url = f"/runs/{run_id}/episodes/{episode_id}"
        title = f"Episode {episode['episode_id']}"
        items.append({"title": title, "kind": "Episode", "url": url,
                      "summary": f"Task {episode.get('task_id', 'Unavailable')} · {episode.get('panel_name') or ''}"})
        if episode.get("task_id") is not None:
            items.append({"title": f"Task {episode['task_id']}", "kind": "Task", "url": url,
                          "summary": title})
    for side in ("customer", "service"):
        for strategy_id in run.get("strategies", {}).get(side, {}):
            encoded = quote(str(strategy_id), safe="")
            items.append({"title": f"{side.title()} {strategy_id}", "kind": "Strategy",
                          "url": f"/runs/{run_id}/strategies/{encoded}?side={side}", "summary": "Saved strategy snapshot"})
    for commit in run.get("generation_commits", ()):
        generation = commit.get("generation")
        items.append({"title": f"Generation {generation}", "kind": "Generation",
                      "url": f"/runs/{run_id}#generation-{generation}", "summary": str(commit.get("note", ""))})
    return items


def _command_index_for_summaries(runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"title": str(item["experiment_id"]), "kind": str(item["phase"]),
             "url": f"/runs/{quote(str(item['run_id']), safe='')}", "summary": str(item["status"])}
            for item in runs]


def _health_view(run: dict[str, Any]) -> list[dict[str, str]]:
    budget = run.get("budget", {})
    provider = (
        f"{budget['successes']} successful / {budget['attempts']} attempts"
        if type(budget.get("successes")) is int and type(budget.get("attempts")) is int
        else "Unavailable"
    )
    return [
        {"label": "Provider success", "value": provider, "state": "ok" if provider != "Unavailable" and budget.get("failures") == 0 else "neutral"},
        {"label": "Heldout", "value": "Sealed" if run.get("heldout_sealed") else "Open / none", "state": "sealed" if run.get("heldout_sealed") else "neutral"},
        {"label": "Artifacts", "value": "Validated for display", "state": "ok"},
        {"label": "Communication mode", "value": run.get("communication_mode_label", "Unknown"), "state": "neutral"},
    ]


def _human_events(run: dict[str, Any]) -> list[dict[str, str]]:
    labels = {
        "run_started": "实验 run 已登记", "run_finished": "实验 run 已完成", "run_failed": "实验 run 中断",
        "generation_committed": "Generation 已提交", "episode_started": "Episode 开始",
        "episode_finished": "Episode 完成", "evaluation_finished": "τ-bench 任务评估完成",
        "tool_call": "Service 调用工具", "tool_result": "工具返回结果",
        "service_strategy_proposed": "Service strategy proposal 已生成",
        "service_strategy_selected": "Service strategy 已比较",
        "budget_updated": "Provider budget 更新", "customer_message": "Customer 发言",
        "agent_message": "Service 回复",
    }
    views = []
    for event in run.get("events", ()):
        payload = event.get("payload") or {}
        kind = event.get("event_type", "")
        title = labels.get(kind, "实验记录更新")
        if kind == "generation_committed":
            title = f"Generation {event.get('generation')} 已提交"
        elif kind == "episode_started":
            title = f"Episode 开始 · {event.get('episode_id') or '当前 episode'}"
        elif kind == "episode_finished":
            title = f"Episode 完成 · Task {payload.get('task_id', 'Unavailable')}"
        elif kind == "service_strategy_selected":
            title = f"Service proposal {'accepted' if payload.get('accepted') else 'not accepted'}"
        elif kind == "tool_call":
            title = f"调用工具 · {payload.get('name') or payload.get('tool_name') or 'tool'}"
        elif kind == "tool_result":
            title = f"工具返回 · {payload.get('tool_name') or 'tool'}"
        elif kind == "evaluation_finished":
            success = payload.get("task_success")
            title = "τ-bench 任务成功" if success is True else "τ-bench 任务未成功" if success is False else title
        timestamp = str(event.get("timestamp") or "")
        try:
            parsed = datetime.fromisoformat(timestamp)
            local = parsed.astimezone() if parsed.tzinfo else parsed
            time_label = local.strftime("%H:%M")
            sort_time = parsed.timestamp()
        except (ValueError, OverflowError, OSError):
            time_label = "—"
            sort_time = 0.0
        views.append({"event_id": str(event.get("event_id") or ""), "title": title,
                      "time": time_label, "sort_time": sort_time, "kind": kind,
                      "generation": str(event.get("generation") if event.get("generation") is not None else "—"),
                      "episode_id": str(event.get("episode_id") or ""),
                      "url": f"/runs/{quote(str(run['run_id']), safe='')}/episodes/{quote(str(event.get('episode_id')), safe='')}"
                      if event.get("episode_id") else ""})
    ordered = sorted(views, key=lambda item: item["sort_time"])
    for item in ordered:
        item.pop("sort_time", None)
    return ordered


def _compare_runs(run_a: dict[str, Any], run_b: dict[str, Any]) -> dict[str, Any]:
    def panel(manifest: dict[str, Any]):
        panels = manifest.get("task_panels")
        if isinstance(panels, dict):
            return {key: panels.get(key) for key in ("E", "V", "H") if key in panels}
        selection = manifest.get("task_selection")
        if isinstance(selection, dict):
            if isinstance(selection.get("heldout"), str):
                return None
            return {key: selection.get(key) for key in ("evolution", "validation", "heldout") if key in selection}
        keys = ("evolution_task_id", "validation_task_id")
        value = {key: manifest.get(key) for key in keys if manifest.get(key) is not None}
        return value or None

    def budget_cap(manifest: dict[str, Any]):
        return manifest.get("request_budget_cap")

    a, b = run_a["manifest"], run_b["manifest"]
    comparisons = [
        _compatibility_row("Phase", run_a.get("phase"), run_b.get("phase")),
        _compatibility_row("Task panel", panel(a), panel(b)),
        _compatibility_row("Seed", a.get("seed"), b.get("seed")),
        _compatibility_row("Request budget", budget_cap(a), budget_cap(b)),
        _compatibility_row("Model configuration", (a.get("role_models"), a.get("role_model_args")),
                           (b.get("role_models"), b.get("role_model_args"))),
        _compatibility_row("Communication mode", a.get("enforce_communication_protocol"),
                           b.get("enforce_communication_protocol"), require_bool=True),
    ]
    for label, field in [
        ("Service carrier", "evolution"),
        ("V2 activation/search/gate/replay/archive policy", "skill_evolution_v2"),
    ]:
        comparisons.append(
            _compatibility_row(label, a.get(field, "legacy"), b.get(field, "legacy"))
        )

    def native_result(run):
        result = run.get("result")
        if not isinstance(result, dict):
            return "Unavailable"
        stored_rate = result.get("native_success_rate")
        if isinstance(stored_rate, (int, float)) and not isinstance(stored_rate, bool):
            return f"{stored_rate:.1%} (stored)"
        reward = result.get("native_reward")
        if isinstance(reward, (int, float)) and not isinstance(reward, bool):
            return f"native reward {reward:g} (stored)"
        episodes = run.get("episodes", ())
        observed = [item.get("task_success") for item in episodes if type(item.get("task_success")) is bool]
        if observed:
            return f"{sum(observed)}/{len(observed)} visible episodes successful"
        return "Unavailable"

    def model_label(manifest):
        models = manifest.get("role_models")
        if not isinstance(models, dict):
            return "Unavailable"
        return ", ".join(f"{role}: {model}" for role, model in sorted(models.items()))

    def protocol_label(manifest):
        value = manifest.get("enforce_communication_protocol")
        return (
            "Strict diagnostic"
            if value is True
            else "Upstream default"
            if value is False
            else "Unknown"
        )

    rows = [
        ("Native task success", native_result(run_a), native_result(run_b)),
        ("Visible episodes", str(len(run_a.get("episodes", ()))), str(len(run_b.get("episodes", ())))),
        ("Provider attempts", run_a["budget"].get("attempt_label", "Unavailable"), run_b["budget"].get("attempt_label", "Unavailable")),
        ("Prompt / completion tokens", run_a["budget"].get("token_label", "Unavailable"), run_b["budget"].get("token_label", "Unavailable")),
        ("Duration", "Unavailable", "Unavailable"),
        ("Model", model_label(a), model_label(b)),
        ("Communication mode", protocol_label(a), protocol_label(b)),
    ]

    def recorded(run, key):
        latest = (run.get("generation_commits") or [{}])[-1]
        candidates = latest.get("service_phase", {}).get("candidates", [])
        selected = next((c for c in candidates if c.get("runtime_deployed")), None)
        if key == "skill_count":
            return latest.get("service_after", {}).get("skill_count", "Unavailable")
        if key == "final_accuracy":
            return latest.get("service_phase", {}).get("final_accuracy", "Unavailable")
        if key == "activation_rate":
            return latest.get("activation_summary", {}).get(
                "activation_rate", "Unavailable"
            )
        if key == "skill_tokens":
            return latest.get("activation_summary", {}).get(
                "rendered_skill_tokens", "Unavailable"
            )
        if key == "gate_confidence":
            return latest.get("statistical_policy", {}).get("confidence", "Unavailable")
        if key == "activator_calls":
            return (
                (run.get("result") or {})
                .get("api_usage_by_role", {})
                .get("skill_activator", {})
                .get("calls", "Unavailable")
            )
        if selected is None:
            return "Unavailable"
        return selected.get("effect", {}).get(key, "Unavailable")

    for label, key in [
        ("Final E Service accuracy", "final_accuracy"),
        ("Activation rate", "activation_rate"),
        ("Rendered skill tokens", "skill_tokens"),
        ("Gate confidence", "gate_confidence"),
        ("Helpfulness", "helpfulness"),
        ("Harmfulness", "harmfulness"),
        ("Stuck rate", "new_stuck_rate"),
        ("Active skills", "skill_count"),
        ("Runtime token overhead", "token_delta"),
        ("Activator calls", "activator_calls"),
    ]:
        rows.append((label, recorded(run_a, key), recorded(run_b, key)))
    comparable = all(item["same"] is True for item in comparisons)
    return {"compatibility": comparisons, "comparable": comparable, "rows": rows}


def _compatibility_row(label: str, left: Any, right: Any, *, require_bool: bool = False) -> dict[str, Any]:
    known = left is not None and right is not None
    if require_bool:
        known = type(left) is bool and type(right) is bool
    same = bool(known and left == right)
    if same:
        detail = "Matched"
    elif not known:
        detail = "Unknown or incomplete frozen metadata"
    else:
        detail = "Different"
    return {
        "label": label,
        "left": _compatibility_label(label, left),
        "right": _compatibility_label(label, right),
        "same": same,
        "detail": detail,
    }


def _compatibility_label(label: str, value: Any) -> str | None:
    if value is None:
        return None
    if label == "Task panel" and isinstance(value, dict):
        names = {"evolution": "E", "validation": "V", "heldout": "H",
                 "evolution_task_id": "E", "validation_task_id": "V"}
        parts = []
        for key, tasks in value.items():
            if isinstance(tasks, (list, tuple)):
                preview = ", ".join(str(task) for task in tasks[:3])
                suffix = f", +{len(tasks) - 3}" if len(tasks) > 3 else ""
                content = f"{len(tasks)} task(s)" + (f" ({preview}{suffix})" if preview else "")
            else:
                content = str(tasks)
            parts.append(f"{names.get(key, key)}: {content}")
        return " · ".join(parts)
    if label == "Model configuration" and isinstance(value, tuple) and len(value) == 2:
        models, args = value
        if not isinstance(models, dict) or not models:
            return "Unknown"
        model_values = list(models.values())
        if all(model == model_values[0] for model in model_values[1:]):
            model_text = f"{model_values[0]} (all roles)"
        else:
            model_text = "; ".join(f"{role}: {model}" for role, model in sorted(models.items()))
        if not isinstance(args, dict):
            return model_text
        display_keys = ("temperature", "top_p", "max_tokens", "max_completion_tokens", "thinking_mode")
        settings: list[str] = []
        for key in display_keys:
            values = [(role, role_args[key]) for role, role_args in args.items()
                      if isinstance(role_args, dict) and key in role_args]
            if not values:
                continue
            display_name = {"max_tokens": "max tokens", "max_completion_tokens": "max completion"}.get(key, key)
            common = all(item == values[0][1] for _, item in values[1:])
            if common and len(values) == len(args):
                settings.append(f"{display_name} {values[0][1]}")
            else:
                settings.extend(f"{role} {display_name} {item}" for role, item in values)
        return model_text + (" · " + " · ".join(settings) if settings else "")
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if type(value) is bool:
        return "Enabled" if value else "Disabled"
    return str(value)


def _artifact_inventory(run: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(run["path"])
    result_name = "phase0-result.json" if run["phase"] == "0-integration-proof" else "alternating-result.json"
    allowed = ["manifest.json", "run-context.json", result_name, "events.jsonl"]
    entries: dict[str, dict[str, Any]] = {}
    def add(path: Path, label: str, state: str = "Available"):
        try:
            if path.is_symlink() or not path.is_file():
                return
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
        except (OSError, ValueError):
            return
        entries[relative] = {"name": label, "path": relative,
                             "sha256": hashlib.sha256(data).hexdigest(),
                             "size": len(data), "status": state}
    for name in allowed:
        path = root / name
        if run.get("heldout_sealed") and name == result_name:
            if path.exists():
                entries[name] = {"name": name, "path": "[sealed]", "sha256": None,
                                 "size": None, "status": "Heldout sealed"}
            continue
        add(path, name, "Console journal" if name == "events.jsonl" else "Read and verified")
    if not run.get("heldout_sealed"):
        for episode in run.get("episodes", ()):
            ref = (episode.get("raw_record") or {}).get("trajectory_ref")
            trajectory_root = episode.get("trajectory_root")
            if not isinstance(ref, str) or not isinstance(trajectory_root, Path):
                continue
            try:
                parent = (trajectory_root / ref).resolve(strict=True).parent
                for filename in (
                    "episode-record.json", "run-telemetry.json",
                    "native-simulation.json", "db-state-trace.json",
                ):
                    if filename == "db-state-trace.json":
                        trace = episode.get("db_state_trace")
                        provenance = trace.get("provenance") if isinstance(trace, dict) else None
                        if not isinstance(provenance, dict) or not provenance.get("trajectory_sha256"):
                            continue
                        artifact_state = "Verified by trace reader"
                    else:
                        artifact_state = (
                            "Verified by run index"
                            if filename == "native-simulation.json"
                            else "Available"
                        )
                    add(parent / filename, filename, artifact_state)
            except (OSError, RuntimeError):
                continue
    return sorted(entries.values(), key=lambda item: item["path"])


async def _read_form(request: Request) -> dict[str, str]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/x-www-form-urlencoded":
        raise HTTPException(status_code=415, detail="expected an HTML form")
    raw = await request.body()
    if len(raw) > 32_768:
        raise HTTPException(status_code=413, detail="form is too large")
    try:
        from urllib.parse import parse_qs

        values = parse_qs(raw.decode("utf-8"), keep_blank_values=True, max_num_fields=40)
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="form is malformed") from exc
    return {key: items[-1] for key, items in values.items() if items}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--tau2-data-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        address = ipaddress.ip_address(args.host)
    except ValueError:
        print("EvoTau Console binds to a loopback IP address only.", file=sys.stderr)
        return 2
    if not address.is_loopback:
        print("EvoTau Console binds to a loopback IP address only.", file=sys.stderr)
        return 2
    try:
        import uvicorn
    except ImportError:
        print('Install the optional web dependencies with: pip install -e ".[web]"', file=sys.stderr)
        return 2
    app = create_app(
        project_root=args.project_root,
        runs_root=args.runs_dir,
        tau2_data_dir=args.tau2_data_dir,
    )
    uvicorn.run(app, host=address.compressed, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
