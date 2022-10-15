# python3
# func: add mlflow
# ==========================================
from typing import Any, Dict
import statsd
import mlflow
import sys
from kedro.framework.hooks import hook_impl
from kedro.pipeline.node import Node


class ModelTrackingHooks:
    # https://kedro.readthedocs.io/en/stable/hooks/examples.html#add-memory-consumption-tracking
    """Namespace for grouping all model-tracking hooks with MLflow together."""
    def __init__(self):
        self._timers = {}
        self._client = statsd.StatsClient(prefix="kedro")

    @hook_impl
    def before_pipeline_run(self, run_params: Dict[str, Any]) -> None:
        """Hook implementation to start an MLflow run
        with the session_id of the Kedro pipeline run.
        """
        mlflow.start_run(run_name=run_params["session_id"], nested=True)
        for k, v in run_params.items():
            if v is None: continue
            if len(v):
                mlflow.log_params({k:v})

    @hook_impl
    def after_node_run(
        self, node: Node, outputs: Dict[str, Any], inputs: Dict[str, Any]
    ) -> None:
        """Hook implementation to add model tracking after some node runs.
        In this example, we will:
        * Log the parameters after the data splitting node runs.
        * Log the model after the model training node runs.
        * Log the model's metrics after the model evaluating node runs.
        """
        for k, v in inputs.items():
            if v is None: continue
            if len(v):
                mlflow.log_params({k:v})

    @hook_impl
    def after_pipeline_run(self) -> None:
        """Hook implementation to end the MLflow run
        after the Kedro pipeline finishes.
        """
        self._client.incr("run")
        mlflow.end_run()


    @hook_impl
    def before_node_run(self, node: Node) -> None:
        node_timer = self._client.timer(node.name)
        node_timer.start()
        self._timers[node.short_name] = node_timer


    @hook_impl
    def after_node_run(self, node: Node, inputs: Dict[str, Any]) -> None:
        self._timers[node.short_name].stop()
        for dataset_name, dataset_value in inputs.items():
            self._client.gauge(dataset_name + "_size", sys.getsizeof(dataset_value))
