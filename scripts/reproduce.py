"""One-command deterministic reviewer workflow.

Authorized-data runs create artifacts but do not claim that missing controlled
audit inputs exist. ``--smoke`` is an explicitly non-evidentiary software test.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMED_ABLATIONS = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "PhasePNoIPCW",
    "PhasePNoCheckpoint",
    "PhasePNoPlatt",
]


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic P-HLPL-HCC reproduction workflow")
    parser.add_argument("--data", default=None, help="Authorized cohort table")
    parser.add_argument("--output-root", default="outputs/reproduction")
    parser.add_argument("--smoke", action="store_true", help="Non-evidentiary reduced test run")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--include-discrete-time", action="store_true")
    args = parser.parse_args(argv)
    if not args.smoke and not args.data:
        raise ValueError("--data is mandatory unless --smoke is explicitly selected")
    python = sys.executable
    if not args.skip_tests:
        run([python, "-m", "unittest", "discover", "-s", "tests", "-v"])
    output_root = Path(args.output_root)
    data_path = args.data
    if args.smoke and not data_path:
        fixture_path = output_root / "fixture_hcc.csv"
        run(
            [
                python,
                "-m",
                "p_hlpl_hcc.data",
                "make-fixture",
                "--out",
                str(fixture_path),
                "--n",
                "120",
                "--seed",
                "42",
            ]
        )
        data_path = str(fixture_path)
    base = [
        python,
        "-m",
        "p_hlpl_hcc.train",
        "--config",
        "configs/default.yaml",
        "--data",
        str(data_path),
    ]
    if args.smoke:
        base.append("--fast")
    run(base + ["--ablation", "full", "--output", str(output_root / "full")])
    if args.include_ablations:
        for name in NAMED_ABLATIONS:
            run(base + ["--ablation", name, "--output", str(output_root / name.lower())])
    if args.include_discrete_time:
        discrete = [
            python,
            "-m",
            "p_hlpl_hcc.train",
            "--config",
            "configs/discrete_time.yaml",
            "--output",
            str(output_root / "discrete_time"),
        ]
        discrete.extend(["--data", str(data_path)])
        if args.smoke:
            discrete.append("--fast")
        run(discrete)
    run([python, "scripts/check_acceptance_artifacts.py", "--root", "."])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
