"""Script principal para execução do pipeline completo."""

import logging
from pathlib import Path

import pandas as pd

from credit_risk_quantum_ml.src import (
    data_pipeline,
    model_training,
    model_evaluation,
    explainability,
    utils,
)


def main() -> None:
    utils.setup_logging()

    # Geração e tratamento dos dados
    df = data_pipeline.generate_synthetic_data()
    df = data_pipeline.clean_data(df)
    train_df, val_df, test_df = data_pipeline.split_data(df)

    X_train = train_df.drop("inadimplente", axis=1).to_numpy()
    y_train = train_df["inadimplente"].to_numpy()
    X_test = test_df.drop("inadimplente", axis=1).to_numpy()
    y_test = test_df["inadimplente"].to_numpy()

    # Treino modelos quânticos
    qsvc = model_training.train_qsvc(X_train, y_train)
    vqc = model_training.train_vqc(X_train, y_train)
    qnn = model_training.train_qnn(X_train, y_train)

    # Treino modelos clássicos
    classical_models = model_training.train_classical_models(X_train, y_train)

    # Avaliação simples usando conjunto de teste
    for name, model in classical_models.items():
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        metrics = model_evaluation.evaluate_model(y_test, y_pred, y_score)
        logging.info("Métricas %s: %s", name, metrics)
        model_evaluation.plot_confusion_matrix(y_test, y_pred, f"Confusion {name}")
        model_evaluation.plot_roc_curve(y_test, y_score, f"ROC {name}")

    # Análise de explicabilidade do feature map do QSVC
    feature_map = data_pipeline.get_feature_map(X_train.shape[1])
    explainability.explain_quantum_feature_map(feature_map)

    # Exemplo de explicação com SHAP para o modelo clássico
    explainability.explain_classical_model(classical_models["random_forest"], X_test)


if __name__ == "__main__":
    main()
