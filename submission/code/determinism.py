"""Deterministic execution harness for the RGTC pipeline.

Calling :func:`seed_everything` before any experiment fixes every source of
run-to-run variation so that the full pipeline regenerates byte-identical
results, figures and tables on every run (CPU, macOS). This is the operational
basis for the reproducibility claim: ``lieideal-verify`` runs the pipeline twice
and asserts the SHA-256 of every output file is unchanged.
"""
from __future__ import annotations

import os
import random


def seed_everything(seed: int = 42) -> int:
    """Fix all RNGs and remove nondeterministic floating-point/encoding sources."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Single-threaded numerical libraries => fixed floating-point reduction order.
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"
    # Reproducible matplotlib PDF output: fixes the embedded creation timestamp.
    # Must match the value used in ``common.py`` so figures are byte-identical
    # regardless of which entry point imports first.
    os.environ["SOURCE_DATE_EPOCH"] = "1700000000"

    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)
    except Exception:
        pass
    return seed
