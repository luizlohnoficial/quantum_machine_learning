"""Módulo de geração e tratamento de dados para risco de crédito."""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qiskit.circuit.library import ZFeatureMap


RANDOM_SEED = 42


def generate_synthetic_data(num_samples: int = 10000) -> pd.DataFrame:
    """Gera dados sintéticos simulando risco de crédito.

    Os atributos seguem parâmetros comumente utilizados pelo mercado,
    como renda, pontuação de crédito e histórico de atrasos.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    data = {
        "idade": rng.integers(18, 70, num_samples),
        "tempo_residencia": rng.integers(0, 40, num_samples),
        "tempo_emprego": rng.integers(0, 30, num_samples),
        "renda": rng.normal(5000, 1500, num_samples).clip(300, None),
        "divida_renda": rng.uniform(0, 1, num_samples),
        "quant_emprestimos": rng.integers(0, 10, num_samples),
        "consultas_spc": rng.integers(0, 5, num_samples),
        "atrasos_passados": rng.integers(0, 10, num_samples),
        "score_bureau": rng.integers(300, 851, num_samples),
        "utilizacao_credito": rng.uniform(0, 1, num_samples),
        "num_dependentes": rng.integers(0, 6, num_samples),
    }
    df = pd.DataFrame(data)
    prob_default = np.clip(
        0.3 * df["divida_renda"]
        + 0.05 * df["consultas_spc"]
        + 0.1 * (df["atrasos_passados"] / 10)
        + 0.2 * (1 - df["score_bureau"] / 850)
        + 0.1 * df["utilizacao_credito"]
        + 0.05 * (df["num_dependentes"] / 5),
        0,
        1,
    )
    df["inadimplente"] = rng.binomial(1, prob_default)
    logging.info("Dados sintéticos gerados: %s amostras", num_samples)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers simples e padroniza colunas numéricas."""
    df = df.drop_duplicates().reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    logging.info("Dados limpos e padronizados")
    return df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Realiza o split em treino, validação e teste."""
    X = df.drop("inadimplente", axis=1)
    y = df["inadimplente"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
    )
    logging.info("Split realizado: treino %s, validacao %s, teste %s", len(X_train), len(X_val), len(X_test))
    return (
        pd.concat([X_train, y_train], axis=1),
        pd.concat([X_val, y_val], axis=1),
        pd.concat([X_test, y_test], axis=1),
    )


def get_feature_map(num_features: int) -> ZFeatureMap:
    """Retorna um Feature Map quântico simples."""
    return ZFeatureMap(feature_dimension=num_features, reps=2)


