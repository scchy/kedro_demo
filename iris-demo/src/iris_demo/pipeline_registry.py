"""Project pipelines."""
from typing import Dict
from iris_demo.pipelines.training_pipline.train_pip import create_pipeline
from iris_demo.pipelines.metric_pipline.metric_pip import create_pipeline as metric_create
from kedro.pipeline import Pipeline, pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    data_science_pipeline = create_pipeline()
    metric_pip_ = metric_create()
    return {
        "__default__": data_science_pipeline + metric_pip_,
        "ds": data_science_pipeline,
        'metric': metric_pip_
    }