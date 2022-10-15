# python3
# Create date: 2022-10-15
# Author: Scc_hy
# Func: 模型评估 pipeline
# ===============================================================================


from .model_metric import evaluate_model
from kedro.pipeline import pipeline, node

def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func=evaluate_model,
                inputs=["logistic_model_v1", "X_test", "y_test"],
                outputs=['metric_info', 'metric_pics'],
                name="evaluate_model_node"
            )
        ]
    )
