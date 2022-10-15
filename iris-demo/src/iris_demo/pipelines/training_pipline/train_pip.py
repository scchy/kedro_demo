# python3
# Create date: 2022-09-05
# Author: Scc_hy
# Func: 模型训练 pipeline
# ===============================================================================

from kedro.pipeline import pipeline, node
from .train_func_node import split_data, train_model


def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["irir_data", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="logistic_model_v1",
                name="train_model_node",
            ),
        ]
    )
