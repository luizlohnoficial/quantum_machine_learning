"""Circuitos quânticos utilizados nos modelos."""

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel


def create_feature_map(num_features: int) -> ZZFeatureMap:
    """Cria um feature map ZZ."""
    return ZZFeatureMap(feature_dimension=num_features, reps=2)


def create_variational_circuit(num_qubits: int) -> TwoLocal:
    """Cria um circuito variacional básico."""
    return TwoLocal(num_qubits, rotation_blocks="ry", entanglement_blocks="cz", reps=3)


def create_quantum_kernel(feature_map: QuantumCircuit) -> QuantumKernel:
    """Retorna um QuantumKernel para uso no QSVC."""
    return QuantumKernel(feature_map=feature_map)
