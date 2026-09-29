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
        return templates.TemplateResponse(
            request=request,
            name=template,
            context={"request": request, **(context or {})},
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
            failure = next((item for item in run["episodes"] if item["episode_id"] == episode_id), None)
            provisional = failure is not None
        if failure is None:
            raise HTTPException(status_code=404, detail="failure unavailable")
        state = failure_status({"verified": not provisional, "provisional": provisional})
        return page(request, "failure.html", {
            "run": run, "failure": failure, "provisional": provisional, "state": state,
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
        return page(request, "crossplay.html", {
            "run": view["run"], "matrices": view["matrices"], "selected": view["selected"],
            "metric": metric, "metric_label": _METRIC_LABELS[metric],
            "cells": _crossplay_cells(matrix, metric),
        })

    return app


def _crossplay_cells(matrix: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    rows = []
    higher_is_better = metric == "native_success_rate"
    for cell in matrix["cells"]:
        value = cell.get(metric)
        if value is None:
            style = "unknown"
        elif value == 0.5:
            style = "mid"
        elif (value > 0.5) == higher_is_better:
            style = "good"
        else:
            style = "bad"
        rows.append({
            **cell,
            "metric_value": value,
            "style": style,
        })
    return rows


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
