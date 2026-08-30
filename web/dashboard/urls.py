from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/status/", views.api_status, name="api_status"),
    path("api/generate/", views.api_generate, name="api_generate"),
    path("api/run/sequential/", views.api_run_sequential, name="api_run_sequential"),
    path("api/run/random/", views.api_run_random, name="api_run_random"),
    path("api/train/", views.api_train, name="api_train"),
    path("api/run/smart/", views.api_run_smart, name="api_run_smart"),
    path("api/compare/", views.api_compare, name="api_compare"),
    path("api/demo-sweep/", views.api_demo_sweep, name="api_demo_sweep"),
    path("api/reset/", views.api_reset, name="api_reset"),
    path("api/viewport/", views.api_viewport, name="api_viewport"),
    path("api/charts/environment/", views.api_charts_environment, name="api_charts_environment"),
    path("api/charts/receiver/", views.api_charts_receiver, name="api_charts_receiver"),
    path("api/charts/ml/", views.api_charts_ml, name="api_charts_ml"),
    path("api/charts/smart/", views.api_charts_smart, name="api_charts_smart"),
    path("api/charts/comparison/", views.api_charts_comparison, name="api_charts_comparison"),
    path("api/emitters/", views.api_emitters, name="api_emitters"),
    path("api/download/environment/", views.api_download_environment, name="api_download_environment"),
    path("api/download/observations/", views.api_download_observations, name="api_download_observations"),
    path("api/download/comparison/", views.api_download_comparison, name="api_download_comparison"),
]
