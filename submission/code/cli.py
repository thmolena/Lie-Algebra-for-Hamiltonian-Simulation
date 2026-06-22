"""Command-line entry points for the RGTC reproducibility package.

``lieideal-reproduce`` regenerates every dataset, figure and table the
manuscript depends on, from first principles, deterministically.
``lieideal-verify`` runs the pipeline twice and asserts that every output file
is byte-for-byte identical across runs -- the operational proof that the study
reproduces exactly on every run.
"""
from __future__ import annotations

import hashlib
import json
import sys

from determinism import seed_everything


def _run_pipeline() -> None:
    import make_all

    # ``make_all`` already runs the dense-matrix experiments, the learned-residual
    # operator-learning experiment, and the headline comparison figure in order.
    make_all.main(force=True)


def main(argv: list[str] | None = None) -> int:
    seed_everything(42)
    _run_pipeline()
    print("rgtc: reproduction complete -- datasets, figures and tables regenerated.")
    return 0


def _hash_outputs() -> dict[str, str]:
    """SHA-256 of every scientific output (data + figures).

    Per-run environment metadata (``*.meta.json``: versions, git commit) is
    excluded because it legitimately records the host, not the science.
    """
    import common

    files = []
    files += sorted(common.DATA_DIR.glob("*.csv"))
    files += [p for p in sorted(common.DATA_DIR.glob("*.json")) if not p.name.endswith(".meta.json")]
    files += sorted(common.FIGURE_DIR.glob("*.pdf"))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def verify(argv: list[str] | None = None) -> int:
    seed_everything(42)
    _run_pipeline()
    first = _hash_outputs()
    _run_pipeline()
    second = _hash_outputs()

    mismatched = {
        name: (first.get(name), second.get(name))
        for name in sorted(set(first) | set(second))
        if first.get(name) != second.get(name)
    }
    if mismatched:
        print("NON-DETERMINISTIC outputs detected:")
        print(json.dumps(mismatched, indent=2))
        return 1
    print(
        f"DETERMINISTIC: all {len(first)} scientific output files "
        f"(datasets + figures) are byte-for-byte identical across two runs."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
