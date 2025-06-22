"""Treinamento de modelos quânticos e clássicos."""

import logging
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.svm import SVC
# `qiskit.algorithms` foi descontinuado nas versões mais recentes do Qiskit e
# os otimizadores passaram a residir no pacote separado `qiskit_algorithms`.
# Tentamos importar primeiro desse caminho e, caso não exista (ambientes com
# Qiskit mais antigo), caímos para o caminho anterior para manter
# compatibilidade.
try:
    from qiskit_algorithms.optimizers import COBYLA  # type: ignore
except ImportError:  # pragma: no cover - depende da versão instalada
    from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import QSVC, VQC
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
import torch

from .quantum_circuits import (
    create_feature_map,
    create_quantum_kernel,
    create_variational_circuit,
)
from .classical_models import (
    get_gradient_boosting,
    get_logistic_regression,
    get_random_forest,
)


def train_qsvc(X: np.ndarray, y: np.ndarray) -> QSVC:
    feature_map = create_feature_map(X.shape[1])
    kernel = create_quantum_kernel(feature_map)
    model = QSVC(quantum_kernel=kernel)
    model.fit(X, y)
    logging.info("QSVC treinado")
    return model


def train_vqc(X: np.ndarray, y: np.ndarray) -> VQC:
    feature_map = create_feature_map(X.shape[1])
    ansatz = create_variational_circuit(X.shape[1])
    optimizer = COBYLA(maxiter=100)
    model = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
    )
    model.fit(X, y)
    logging.info("VQC treinado")
    return model


def train_qnn(X: np.ndarray, y: np.ndarray) -> TorchConnector:
    feature_map = create_feature_map(X.shape[1])
    ansatz = create_variational_circuit(X.shape[1])
    qnn = EstimatorQNN(feature_map=feature_map, ansatz=ansatz)
    model = TorchConnector(qnn)
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    for _ in range(30):
        optimizer.zero_grad()
        output = model(X_t)
        l = loss(output.squeeze(), y_t)
        l.backward()
        optimizer.step()
    logging.info("QNN treinado")
    return model


def train_classical_models(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    models = {
        "logistic_regression": get_logistic_regression(),
        "random_forest": get_random_forest(),
        "gradient_boosting": get_gradient_boosting(),
    }
    for name, model in models.items():
        model.fit(X, y)
        logging.info("Modelo %s treinado", name)
    return models
