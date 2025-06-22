"""Avaliação de modelos de risco de crédito."""

import logging
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Calcula métricas de avaliação."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks = max(tpr - fpr)
    metrics = {
        "auc": auc(fpr, tpr),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "ks": ks,
    }
    logging.info("Avaliação realizada: %s", metrics)
    return metrics


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, title: str) -> None:
    """Plota curva ROC."""
    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.title(title)
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    """Plota matriz de confusão."""
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(title)
    plt.show()
