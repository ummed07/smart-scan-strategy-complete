"""Django views for the Smart Scan Strategy dashboard."""

from __future__ import annotations

import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from . import services
from .store import load_state


from .json_utils import sanitize_for_json


def index(request):
    return render(request, "dashboard/index.html")


def _parse_json(request) -> dict:
    if not request.body:
        return {}
    return json.loads(request.body.decode("utf-8"))


def _json_ok(data: dict, status: int = 200) -> JsonResponse:
    return JsonResponse(sanitize_for_json(data), status=status)


@require_GET
def api_status(request):
    return _json_ok(services.get_status(request))


@require_POST
def api_generate(request):
    try:
        payload = _parse_json(request)
        return _json_ok(services.generate_environment(request, payload))
    except Exception as exc:
        return _json_ok({"ok": False, "message": str(exc)}, status=400)


@require_POST
def api_run_sequential(request):
    try:
        return _json_ok(services.run_sequential(request))
    except ValueError as exc:
        return _json_ok({"ok": False, "message": str(exc)}, status=400)


@require_POST
def api_run_random(request):
    try:
        return _json_ok(services.run_random(request))
    except ValueError as exc:
        return _json_ok({"ok": False, "message": str(exc)}, status=400)


@require_POST
def api_train(request):
    return _json_ok(services.train_model(request))


@require_POST
def api_run_smart(request):
    return _json_ok(services.run_smart(request))


@require_POST
def api_compare(request):
    return _json_ok(services.run_comparison(request))


@require_POST
def api_demo_sweep(request):
    try:
        return _json_ok(services.run_demo_sweep(request))
    except ValueError as exc:
        return _json_ok({"ok": False, "message": str(exc)}, status=400)


@require_POST
def api_reset(request):
    return _json_ok(services.reset_session(request))


@require_POST
def api_viewport(request):
    payload = _parse_json(request)
    return _json_ok(services.update_viewport(request, payload))


@require_GET
def api_charts_environment(request):
    state = load_state(request)
    fig = services.chart_environment(state)
    return _json_ok({"chart": fig})


@require_GET
def api_charts_receiver(request):
    state = load_state(request)
    return _json_ok({
        "chart": services.chart_receiver(state),
        "outcomes": services.chart_outcomes(state),
        "history": services.history_table(state),
    })


@require_GET
def api_charts_ml(request):
    state = load_state(request)
    return _json_ok({"charts": services.chart_ml(state)})


@require_GET
def api_charts_smart(request):
    state = load_state(request)
    charts = services.chart_smart(state)
    history = services.history_table(state, source="smart")
    return _json_ok({"charts": charts, "history": history})


@require_GET
def api_charts_comparison(request):
    state = load_state(request)
    return _json_ok({"data": services.chart_comparison(state)})


@require_GET
def api_emitters(request):
    state = load_state(request)
    return _json_ok({"rows": services.emitter_table(state)})


@require_GET
def api_download_environment(request):
    try:
        return services.download_environment_csv(load_state(request))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)


@require_GET
def api_download_observations(request):
    try:
        return services.download_observations_csv(load_state(request))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)


@require_GET
def api_download_comparison(request):
    try:
        return services.download_comparison_csv(load_state(request))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
