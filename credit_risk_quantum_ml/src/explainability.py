"""Explicabilidade dos modelos."""

import logging
from typing import Any

import matplotlib.pyplot as plt
import shap


shap.initjs()


def explain_classical_model(model: Any, X: Any) -> None:
    """Gera explicações SHAP para modelos clássicos."""
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X)
    logging.info("Explicabilidade gerada para modelo clássico")


def explain_quantum_feature_map(feature_map: Any) -> None:
    """Analisa pesos das features no Feature Map."""
    print(feature_map.draw(output="text"))
    logging.info("Feature map analisado")
