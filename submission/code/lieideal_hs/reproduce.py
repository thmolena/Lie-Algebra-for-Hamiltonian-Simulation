"""Console entry point for deterministic regeneration of submission artifacts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _code_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(script: str, *args: str) -> None:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run([sys.executable, script, *args], cwd=_code_dir(), env=env, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute all cached dense-matrix artifacts")
    parser.add_argument("--skip-validate", action="store_true", help="do not run validate_submission.py")
    args = parser.parse_args(argv)

    cmd = ["--force"] if args.force else []
    _run("make_all.py", *cmd)
    if not args.skip_validate:
        _run("validate_submission.py")


if __name__ == "__main__":
    main()
