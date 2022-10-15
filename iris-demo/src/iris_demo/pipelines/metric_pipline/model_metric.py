# python3
# Create date: 2022-09-05
# Author: Scc_hy
# Func: 模型评估
# =============================================================================

import mlflow
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import logging

log_ = logging.getLogger(__name__)

def conf_heat_map(conf_matrix):
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    sns.heatmap(conf_matrix, ax=axes, annot=True, vmin=conf_matrix.min()-1, 
            vmax=conf_matrix.max() + conf_matrix.min())
    axes.set_title('heatmap')
    return fig
    

def evaluate_model(estimator, X_test, y_test):
    metric_info = {}
    y_pred = estimator.predict(X_test)
    score = f1_score(y_test.values.ravel(), y_pred.ravel(), average='macro')
    conf_matrix = confusion_matrix(y_test.values.ravel(), y_pred.ravel())
    fig = conf_heat_map(conf_matrix)
    log_.info(f"[ valid ] f1-score {score:.3f}")
    metric_info['f1_score'] = score
    metric_info['precision_score'] = precision_score(y_test.values.ravel(), y_pred.ravel(), average='macro')
    metric_info['recall_score'] = recall_score(y_test.values.ravel(), y_pred.ravel(), average='macro')
    metric_info['classification_report'] = classification_report(y_test.values.ravel(), y_pred.ravel())

    mlflow.log_metric(key='f1-score', step=1, value=score)
    return [
        metric_info, {'heatmap.png' : fig}
    ]

