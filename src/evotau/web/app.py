"""Local, server-rendered Experiment Console backed by immutable EvoTau artifacts."""

from __future__ import annotations

import argparse
import asyncio
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
from .view_models import (
    budget_view,
    current_strategy_for_episode,
    customer_strategy_view,
    failure_status,
    generation_view,
    service_strategy_view,
)

_WEB_ROOT = Path(__file__).resolve().parent
_METRIC_LABELS = {
    "verified_failure_rate": "经过验证的客服错误率",
    "native_success_rate": "原生任务成功率",
    "policy_violation_rate": "策略违规率",
    "recurrent_verified_failure_rate": "已修复错误的复发率",
}


def create_app(
    *,
    project_root: str | Path | None = None,
    runs_root: str | Path | None = None,
    tau2_data_dir: str | Path | None = None,
    manager: RunManager | None = None,
) -> FastAPI:
    root = Path(project_root or _WEB_ROOT.parents[2]).expanduser().resolve()
    output_root = Path(runs_root or root / "experiments" / "runs").expanduser().resolve()
    reader = ArtifactReader(output_root, project_root=root)
    run_manager = manager or RunManager(root, output_root, tau2_data_dir=tau2_data_dir)
    journal = EventJournal()
    templates = Jinja2Templates(directory=str(_WEB_ROOT / "templates"))
    app = FastAPI(title="EvoTau Experiment Console", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(_WEB_ROOT / "static")), name="static")
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
        run["generation_views"] = [generation_view(item) for item in run["generation_commits"]]
        run["customer_strategy_views"] = {
            key: customer_strategy_view(value)
            for key, value in run["strategies"]["customer"].items()
        }
        run["service_strategy_views"] = {
            key: service_strategy_view(value)
            for key, value in run["strategies"]["service"].items()
        }
        run["current_customer_view"] = customer_strategy_view(run.get("current_customer_strategy"))
        run["current_service_view"] = service_strategy_view(run.get("current_service_strategy"))
        protocol_mode = run["manifest"].get("enforce_communication_protocol")
        run["communication_mode_label"] = (
            "Strict diagnostic" if protocol_mode is True else
            "Upstream default" if protocol_mode is False else "Unknown (legacy artifact)"
        )
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
            "config_phase3": "configs/phase3-mechanism.yaml",
            "config_pilot": "",
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
        phase0_result_path = form.get("phase0_result_path", "")
        provider_plugin = form.get("provider_plugin", "")
        csrf_token = form.get("csrf_token", "")
        if not await consume_csrf_token(csrf_token):
            raise HTTPException(status_code=403, detail="form token expired; reload the dashboard")
        try:
            preview = run_manager.preview(
                phase=phase,
                config_path=config_path,
                tau2_data_dir=tau2_data_dir or None,
                phase0_result_path=phase0_result_path or None,
                provider_plugin=provider_plugin or None,
            )
        except RunManagerError as exc:
            return page(request, "error.html", {
                "title": "无法生成安全预览", "message": str(exc),
            }, status_code=400)
        return page(request, "preview.html", {
            "preview": preview.to_view(),
            "config_path": config_path,
            "tau2_data_dir": tau2_data_dir,
            "phase0_result_path": phase0_result_path,
            "provider_plugin": provider_plugin,
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
        phase0_result_path = form.get("phase0_result_path", "")
        provider_plugin = form.get("provider_plugin", "")
        confirm_experiment_id = form.get("confirm_experiment_id", "")
        csrf_token = form.get("csrf_token", "")
        if not await consume_csrf_token(csrf_token):
            raise HTTPException(status_code=403, detail="form token expired; reload the preview")
        try:
            preview = run_manager.preview(
                phase=phase, config_path=config_path,
                tau2_data_dir=tau2_data_dir or None,
                phase0_result_path=phase0_result_path or None,
                provider_plugin=provider_plugin or None,
            )
            if not hmac.compare_digest(confirm_experiment_id, preview.experiment_id):
                raise RunManagerError("请准确输入预览中的 Experiment ID 后再启动。")
            started = run_manager.start(
                phase=phase,
                config_path=config_path,
                confirmed_manifest_sha256=confirmed_manifest_sha256,
                confirmed_launch_sha256=form.get("confirmed_launch_sha256", ""),
                tau2_data_dir=tau2_data_dir or None,
                phase0_result_path=phase0_result_path or None,
                provider_plugin=provider_plugin or None,
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
        return page(request, "run.html", {
            "run": run, "pause_available": pause_available,
            "csrf_token": issue_csrf_token(),
        })

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
        failure: str = "", adherence: str = "",
    ):
        try:
            run = load_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="run unavailable") from None
        episode_rows = list(run["episodes"])
        supplied = {
            "generation": generation, "panel": panel, "customer": customer,
            "service": service, "task": task, "seed": seed,
            "failure": failure, "adherence": adherence,
        }
        def matches(item: dict[str, Any]) -> bool:
            for key, value in supplied.items():
                if not value:
                    continue
                if key == "generation" and str(item.get("generation")) != value:
                    return False
                if key == "panel" and str(item.get("panel_name") or "") != value:
                    return False
                if key == "customer" and str(item.get("customer_strategy_id") or "") != value:
                    return False
                if key == "service" and str(item.get("service_strategy_id") or "") != value:
                    return False
                if key == "task" and str(item.get("task_id") or "") != value:
                    return False
                if key == "seed" and str(item.get("seed")) != value:
                    return False
                if key == "failure" and (
                    "verified" if item.get("verified_failure") else
                    "provisional" if item.get("policy_violation") else "none"
                ) != value:
                    return False
                if key == "adherence" and (
                    "yes" if item.get("customer_strategy_adherent") is True else
                    "no" if item.get("customer_strategy_adherent") is False else "unknown"
                ) != value:
                    return False
            return True
        filtered = [item for item in episode_rows if matches(item)]
        options = {
            "generation": sorted({str(item["generation"]) for item in episode_rows if item.get("generation") is not None}),
            "panel": sorted({str(item["panel_name"]) for item in episode_rows if item.get("panel_name")}),
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
            "customer_strategy": customer_strategy_view(customer),
            "service_strategy": service_strategy_view(service),
        })

    @app.get("/runs/{run_id}/failures/{failure_id}", response_class=HTMLResponse)
    async def failure_page(request: Request, run_id: str, failure_id: str):
        try:
            run = load_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="failure unavailable") from None
        failure = next((item for item in run["verified_failures"]
                        if item["failure_id"] == failure_id), None)
        provisional = False
        if failure is None and failure_id.startswith("candidate-"):
            episode_id = failure_id.removeprefix("candidate-")
            failure = next((item for item in run["provisional_failures"]
                            if item["episode_id"] == episode_id), None)
            provisional = failure is not None
        if failure is None:
            raise HTTPException(status_code=404, detail="failure unavailable")
        state = failure_status({"verified": not provisional, "provisional": provisional})
        source_episode = None
        reproduction_episode = None
        if provisional:
            source_episode = failure
        else:
            try:
                source_episode = reader.episode(run_id, failure["episode_id"])
                reproduction_episode = reader.episode(run_id, failure["reproduction_episode_id"])
            except FileNotFoundError:
                pass
        repair_view = _failure_repair_view(run, failure, source_episode) if not provisional else None
        return page(request, "failure.html", {
            "run": run, "failure": failure, "provisional": provisional, "state": state,
            "source_episode": source_episode, "reproduction_episode": reproduction_episode,
            "repair_view": repair_view,
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

    @app.get("/runs/{run_id}/cross-play", response_class=HTMLResponse)
    async def crossplay_page(request: Request, run_id: str, metric: str = "verified_failure_rate", cell: str = ""):
        if metric not in _METRIC_LABELS:
            raise HTTPException(status_code=400, detail="unsupported cross-play metric")
        try:
            view = reader.crossplay(run_id, cell or None)
        except FileNotFoundError:
            try:
                run = load_run(run_id)
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="cross-play unavailable") from None
            return page(request, "crossplay.html", {
                "run": run, "matrices": [], "selected": None,
                "metric": metric, "metric_label": _METRIC_LABELS[metric],
            })
        except ArtifactReadError:
            raise HTTPException(status_code=422, detail="cross-play artifact is invalid") from None
        matrix = view["selected"]["matrix"]
        page_run = load_run(run_id)
        cells = _crossplay_cells(matrix, metric)
        for cell_view in cells:
            cell_view["diagonal"] = (
                matrix["customer_strategy_ids"].index(cell_view["customer_strategy_id"])
                == matrix["service_strategy_ids"].index(cell_view["service_strategy_id"])
            )
            cell_view["episode_ids"] = [
                episode["episode_id"] for episode in page_run["episodes"]
                if episode.get("customer_strategy_id") == cell_view["customer_strategy_id"]
                and episode.get("service_strategy_id") == cell_view["service_strategy_id"]
            ]
        return page(request, "crossplay.html", {
            "run": page_run, "matrices": view["matrices"], "selected": view["selected"],
            "metric": metric, "metric_label": _METRIC_LABELS[metric],
            "cells": cells, "services": matrix["service_strategy_ids"],
            "customers": matrix["customer_strategy_ids"],
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

    @app.get("/runs/{run_id}/task-review", response_class=HTMLResponse)
    async def task_review_page(request: Request, run_id: str):
        try:
            run = load_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="run unavailable") from None
        review = run["manifest"].get("task_semantic_review")
        if run["heldout_sealed"]:
            review_view = {"sealed": True, "label": "Review details sealed", "tasks": None,
                           "pairs": None, "reviewer": None, "timestamp": None, "digest": None}
        elif isinstance(review, dict):
            review_view = {
                "sealed": False, "label": run["task_review"]["label"],
                "tasks": len(review.get("task_reviews", ())),
                "pairs": len(review.get("pairwise_reviews", ())),
                "reviewer": review.get("reviewer_id"), "timestamp": review.get("reviewed_at"),
                "digest": run["manifest"].get("task_semantic_review_sha256"),
            }
        else:
            review_view = {"sealed": False, "label": "未提供", "tasks": 0, "pairs": 0,
                           "reviewer": None, "timestamp": None, "digest": None}
        return page(request, "task_review.html", {"run": run, "review": review_view})

    return app


def _crossplay_cells(matrix: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    rows = []
    components = {
        "verified_failure_rate": ("verified_failure_episodes", "strategy_adherent_episodes", "Verified failures", "Strategy-adherent episodes"),
        "native_success_rate": ("successful_episodes", "valid_episodes", "Native successes", "Valid episodes"),
        "policy_violation_rate": ("policy_violation_episodes", "valid_episodes", "Policy violations", "Valid episodes"),
        "recurrent_verified_failure_rate": ("recurrent_verified_failure_episodes", "strategy_adherent_episodes", "Recurrent verified failures", "Strategy-adherent episodes"),
    }
    numerator_key, denominator_key, numerator_label, denominator_label = components[metric]
    for cell in matrix["cells"]:
        value = cell.get(metric)
        if value is None:
            style = "unknown"
        elif value < 0.25:
            style = "low"
        elif value < 0.60:
            style = "mid"
        else:
            style = "high"
        rows.append({
            **cell,
            "metric_value": value,
            "style": style,
            "metric_numerator": cell.get(numerator_key),
            "metric_denominator": cell.get(denominator_key),
            "numerator_label": numerator_label,
            "denominator_label": denominator_label,
        })
    return rows


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
    for failure in run.get("verified_failures", ()):
        failure_id = quote(str(failure["failure_id"]), safe="")
        items.append({"title": str(failure["failure_id"]), "kind": "Verified failure",
                      "url": f"/runs/{run_id}/failures/{failure_id}",
                      "summary": str((failure.get("signature") or {}).get("mistake_type", ""))})
    for episode in run.get("provisional_failures", ()):
        episode_id = quote(str(episode["episode_id"]), safe="")
        items.append({"title": f"Candidate {episode.get('failure_category') or episode_id}",
                      "kind": "Provisional", "url": f"/runs/{run_id}/failures/candidate-{episode_id}",
                      "summary": "尚未验证，不计入 fitness"})
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
    observations = [
        (episode.get("telemetry") or {}).get("communication_protocol_observation")
        for episode in run.get("episodes", ())
    ]
    counts = [item.get("mixed_text_tool_call_message_count") for item in observations if isinstance(item, dict)]
    mixed_label = (
        f"{sum(counts)} mixed text/tool messages observed"
        if counts else "Unavailable"
    )
    provider = (
        f"{budget['successes']} successful / {budget['attempts']} attempts"
        if type(budget.get("successes")) is int and type(budget.get("attempts")) is int
        else "Unavailable"
    )
    cache = budget.get("cache_hits")
    cache_label = str(cache) if type(cache) is int else "Unavailable"
    retry = run.get("manifest", {}).get("provider_retries")
    retry_label = str(retry) if type(retry) is int else "Unavailable"
    progress = budget.get("progress")
    remaining = f"{max(0, 100 - progress * 100):.1f}% remaining" if isinstance(progress, (int, float)) else "Unavailable"
    return [
        {"label": "Provider success", "value": provider, "state": "ok" if provider != "Unavailable" and budget.get("failures") == 0 else "neutral"},
        {"label": "Retry policy", "value": f"{retry_label} configured", "state": "neutral"},
        {"label": "Cache hits", "value": cache_label, "state": "neutral"},
        {"label": "Heldout", "value": "Sealed" if run.get("heldout_sealed") else "Open / none", "state": "sealed" if run.get("heldout_sealed") else "neutral"},
        {"label": "Artifacts", "value": "Validated for display", "state": "ok"},
        {"label": "Communication mode", "value": run.get("communication_mode_label", "Unknown"), "state": "neutral"},
        {"label": "Mixed text/tool messages", "value": mixed_label, "state": "neutral"},
        {"label": "Budget remaining", "value": remaining, "state": "neutral"},
    ]


def _human_events(run: dict[str, Any]) -> list[dict[str, str]]:
    labels = {
        "run_started": "实验 run 已登记", "run_finished": "实验 run 已完成", "run_failed": "实验 run 中断",
        "generation_committed": "Generation 已提交", "episode_started": "Episode 开始",
        "episode_finished": "Episode 完成", "evaluation_finished": "τ-bench 任务评估完成",
        "tool_call": "Service 调用工具", "tool_result": "工具返回结果",
        "failure_verified": "Service failure 已验证", "service_repair_proposed": "提出 Service repair",
        "service_gate_finished": "Repair gate 完成", "service_repair_accepted": "Repair 已接受",
        "service_repair_rejected": "Repair 被拒绝", "service_repair_inconclusive": "Repair 证据不足",
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
        elif kind == "failure_verified":
            title = f"{payload.get('failure_id', 'Failure')} 已验证为 Service failure"
        elif kind == "service_gate_finished":
            gate_label = {"accepted": "通过", "rejected": "未通过", "inconclusive": "证据不足"}.get(
                payload.get("status"), "状态不可用",
            )
            title = f"Repair gate：{gate_label}"
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
        selection = manifest.get("task_selection")
        if isinstance(selection, dict):
            if isinstance(selection.get("heldout"), str):
                return None
            return {key: selection.get(key) for key in ("evolution", "validation", "heldout") if key in selection}
        keys = ("evolution_task_id", "validation_task_id")
        value = {key: manifest.get(key) for key in keys if manifest.get(key) is not None}
        return value or None

    def seed_schedule(manifest: dict[str, Any]):
        if "evolution_seeds" in manifest:
            return manifest.get("evolution_seeds")
        return manifest.get("seed")

    def budget_cap(manifest: dict[str, Any]):
        return manifest.get("request_budget_cap")

    a, b = run_a["manifest"], run_b["manifest"]
    comparisons = [
        _compatibility_row("Phase", run_a.get("phase"), run_b.get("phase")),
        _compatibility_row("Task panel", panel(a), panel(b)),
        _compatibility_row("Seed schedule", seed_schedule(a), seed_schedule(b)),
        _compatibility_row("Request budget", budget_cap(a), budget_cap(b)),
        _compatibility_row("Model configuration", (a.get("role_models"), a.get("role_model_args")),
                           (b.get("role_models"), b.get("role_model_args"))),
        _compatibility_row("Communication mode", a.get("enforce_communication_protocol"),
                           b.get("enforce_communication_protocol"), require_bool=True),
    ]
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
        return "Unavailable (no frozen run-level summary)"
    def model_label(manifest):
        models = manifest.get("role_models")
        if not isinstance(models, dict):
            return "Unavailable"
        return ", ".join(f"{role}: {model}" for role, model in sorted(models.items()))
    def protocol_label(manifest):
        value = manifest.get("enforce_communication_protocol")
        return "Strict diagnostic" if value is True else "Upstream default" if value is False else "Unknown (legacy artifact)"
    rows = [
        ("Native task success", native_result(run_a), native_result(run_b)),
        ("Verified failures", str(len(run_a.get("verified_failures", ()))), str(len(run_b.get("verified_failures", ())))),
        ("Visible episodes", str(len(run_a.get("episodes", ()))), str(len(run_b.get("episodes", ())))),
        ("Provider attempts", run_a["budget"].get("attempt_label", "Unavailable"), run_b["budget"].get("attempt_label", "Unavailable")),
        ("Prompt / completion tokens", run_a["budget"].get("token_label", "Unavailable"), run_b["budget"].get("token_label", "Unavailable")),
        ("Duration", "Unavailable", "Unavailable"),
        ("Model", model_label(a), model_label(b)),
        ("Communication mode", protocol_label(a), protocol_label(b)),
    ]
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
    import hashlib

    root = Path(run["path"])
    manifest_name = "pilot-manifest.json" if (root / "pilot-manifest.json").exists() else "manifest.json"
    result_name = ("pilot-result.json" if run["phase"].startswith("4-") else
                   "phase0-result.json" if run["phase"].startswith("0-") else "phase3-result.json")
    allowed = [manifest_name, result_name, "archive.sqlite", "events.jsonl",
               "crossplay-matrix.json", "crossplay.json", "crossplay-matrices.json"]
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
        if run.get("heldout_sealed") and name in {result_name, "archive.sqlite"}:
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
                for filename in ("episode-record.json", "run-telemetry.json", "native-simulation.json"):
                    add(parent / filename, filename, "Verified by run index" if filename == "native-simulation.json" else "Available")
            except (OSError, RuntimeError):
                continue
    return sorted(entries.values(), key=lambda item: item["path"])


def _failure_repair_view(
    run: dict[str, Any], failure: dict[str, Any], source_episode: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if source_episode is None:
        return None
    for commit in run.get("generation_commits", ()):
        service = ((commit.get("decision_record") or {}).get("service") or {})
        gate = service.get("gate")
        if not isinstance(gate, dict) or gate.get("target_failure_id") != failure.get("failure_id"):
            continue
        after_id = commit.get("service_id")
        candidates = [
            episode for episode in run.get("episodes", ())
            if episode.get("episode_id") != failure.get("episode_id")
            and episode.get("task_id") == source_episode.get("task_id")
            and episode.get("seed") == source_episode.get("seed")
            and episode.get("customer_strategy_id") == source_episode.get("customer_strategy_id")
            and episode.get("service_strategy_id") == after_id
        ]
        return {
            "generation": commit.get("generation"),
            "accepted": gate.get("accepted") is True,
            "inconclusive": gate.get("inconclusive") is True,
            "before_service_id": (service.get("incumbent_before") or {}).get("strategy_id"),
            "after_service_id": after_id,
            "gate": gate,
            "after_episode": candidates[0] if candidates else None,
        }
    return None


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
