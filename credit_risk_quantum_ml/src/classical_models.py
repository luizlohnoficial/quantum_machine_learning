"""Modelos clássicos para benchmark."""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_logistic_regression() -> LogisticRegression:
    """Retorna modelo de Regressão Logística."""
    return LogisticRegression(max_iter=1000)


def get_random_forest() -> RandomForestClassifier:
    """Retorna modelo Random Forest."""
    return RandomForestClassifier(n_estimators=200, random_state=42)


def get_gradient_boosting() -> GradientBoostingClassifier:
    """Retorna modelo Gradient Boosting."""
    return GradientBoostingClassifier()
