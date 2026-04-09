from __future__ import annotations

import numpy as np

from .models import Array


def operator_spectral_error(target: Array, approx: Array) -> float:
    return float(np.linalg.norm(target - approx, ord=2))


def frobenius_error(target: Array, approx: Array) -> float:
    return float(np.linalg.norm(target - approx, ord="fro"))


def normalized_trace_fidelity(target: Array, approx: Array) -> float:
    dimension = target.shape[0]
    overlap = np.trace(target.conj().T @ approx)
    return float((abs(overlap) ** 2) / (dimension**2))
