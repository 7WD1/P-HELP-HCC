"""Check whether reviewer-requested acceptance artifacts are present.

This script is intentionally conservative. It does not certify scientific
validity; it only checks whether the artifact classes repeatedly requested by
the IEEE IoTJ-style reviewers exist in a local, auditable bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Requirement:
    key: str
    label: str
    patterns: tuple[str, ...]
    minimum: int = 1
    hard_blocker: bool = True


REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement(
        key="split_manifests",
        label="Private/public split manifests",
        patterns=("outputs/**/splits_seed_*.json", "audit/**/splits_seed_*.json", "artifacts/**/splits_seed_*.json"),
        minimum=5,
    ),
    Requirement(
        key="per_fold_metrics",
        label="Per-fold metrics",
        patterns=(
            "outputs/**/metrics.json",
            "outputs/**/*metrics.json",
            "audit/**/metrics.json",
            "audit/**/*metrics.json",
            "artifacts/**/metrics.json",
            "artifacts/**/*metrics.json",
        ),
        minimum=25,
    ),
    Requirement(
        key="per_fold_predictions",
        label="Per-fold predictions or paired prediction tables",
        patterns=(
            "outputs/**/*prediction*.csv",
            "outputs/**/*pred*.parquet",
            "audit/**/*prediction*.csv",
            "artifacts/**/*prediction*.csv",
            "artifacts/**/*pred*.parquet",
        ),
        minimum=25,
    ),
    Requirement(
        key="model_checkpoints",
        label="Model checkpoints or fold models",
        patterns=(
            "outputs/**/model.joblib",
            "outputs/**/*checkpoint*",
            "audit/**/model.joblib",
            "audit/**/*model.joblib",
            "audit/**/*checkpoint*",
            "artifacts/**/model.joblib",
            "artifacts/**/*model.joblib",
            "artifacts/**/*checkpoint*",
            "artifacts/**/*.pt",
            "artifacts/**/*.pth",
        ),
        minimum=25,
    ),
    Requirement(
        key="hashes",
        label="Model/data hashes or checksums",
        patterns=("outputs/**/*hash*.json", "audit/**/*hash*", "artifacts/**/*hash*", "artifacts/**/*checksum*"),
        minimum=1,
    ),
    Requirement(
        key="statistical_scripts",
        label="Statistical-test and bootstrap scripts",
        patterns=(
            "scripts/**/*stat*.py",
            "scripts/**/*bootstrap*.py",
            "scripts/**/*mcnemar*.py",
            "scripts/**/*nadeau*.py",
            "audit/**/*stat*.py",
            "artifacts/**/*stat*.py",
        ),
        minimum=1,
    ),
    Requirement(
        key="figure_inputs",
        label="Figure-generation raw inputs",
        patterns=(
            "figures/**/*.csv",
            "figures/**/*.json",
            "figures/**/*.parquet",
            "figures/**/*.xlsx",
            "figures/**/*.npy",
            "figures/**/*.npz",
            "audit/**/figure_inputs/**/*.csv",
            "audit/**/figure_inputs/**/*.json",
            "audit/**/figure_inputs/**/*.parquet",
            "audit/**/figure_inputs/**/*.xlsx",
            "audit/**/*figure*input*",
            "artifacts/**/figure_inputs/**/*.csv",
            "artifacts/**/figure_inputs/**/*.json",
            "artifacts/**/figure_inputs/**/*.parquet",
            "artifacts/**/figure_inputs/**/*.xlsx",
            "artifacts/**/*figure*input*",
        ),
        minimum=1,
    ),
    Requirement(
        key="censoring_audit",
        label="Censoring tables and 60-month sensitivity outputs",
        patterns=(
            "outputs/**/*censor*",
            "outputs/**/*ipcw*",
            "outputs/**/*60*month*",
            "audit/**/*censor*",
            "audit/**/*ipcw*",
            "audit/**/*60*month*",
            "artifacts/**/*censor*",
            "artifacts/**/*ipcw*",
            "artifacts/**/*60*month*",
        ),
        minimum=1,
    ),
    Requirement(
        key="locked_external_validation",
        label="Locked-model external-validation outputs",
        patterns=(
            "outputs/**/*locked*external*",
            "outputs/**/*external*validation*",
            "outputs/**/*transfer*",
            "audit/**/*external*validation*",
            "artifacts/**/*external*validation*",
            "artifacts/**/*locked*external*",
        ),
        minimum=1,
    ),
    Requirement(
        key="edge_profiling",
        label="Measured edge-device profiling logs",
        patterns=(
            "outputs/**/*profil*",
            "outputs/**/*jetson*",
            "outputs/**/*latency*log*",
            "outputs/**/*power*",
            "audit/**/*profil*",
            "artifacts/**/*profil*",
            "artifacts/**/*jetson*",
            "artifacts/**/*latency*log*",
        ),
        minimum=1,
    ),
    Requirement(
        key="environment_lock",
        label="Environment lockfile or container recipe",
        patterns=("environment.yml", "requirements.txt", "requirements-lock.txt", "Dockerfile", "artifacts/**/*environment*", "audit/**/*environment*"),
        minimum=4,
        hard_blocker=False,
    ),
)


EXPECTED_SEEDS = {42, 123, 2024, 31415, 65537}
PREDICTION_COLUMNS = {
    "patient_id",
    "seed",
    "fold",
    "true_class",
    "event",
    "overall_survival_months",
    "pred_class",
    *{f"p_c{index}" for index in range(1, 9)},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_audit_manifest(root: Path) -> tuple[list[str], list[str]]:
    """Validate the controlled packet manifest, referenced files, and hashes."""

    manifest_path = root / "audit" / "manifest.json"
    if not manifest_path.is_file():
        return ["audit/manifest.json is missing"], []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"audit/manifest.json is unreadable: {exc}"], []
    errors: list[str] = []
    referenced: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if manifest.get("evidentiary_status") != "real_experiment_outputs":
        errors.append("evidentiary_status must be 'real_experiment_outputs'")
    fold_runs = manifest.get("fold_runs")
    if not isinstance(fold_runs, list):
        return errors + ["fold_runs must be a list"], referenced
    pairs: set[tuple[int, int]] = set()
    for index, run in enumerate(fold_runs):
        if not isinstance(run, dict):
            errors.append(f"fold_runs[{index}] must be an object")
            continue
        try:
            seed, fold = int(run["seed"]), int(run["fold"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"fold_runs[{index}] needs integer seed/fold")
            continue
        pairs.add((seed, fold))
        for key in ("metrics", "predictions", "model"):
            rel = run.get(key)
            expected_hash = run.get(f"{key}_sha256")
            if not isinstance(rel, str) or not rel:
                errors.append(f"fold_runs[{index}].{key} is missing")
                continue
            path = (root / rel).resolve()
            try:
                path.relative_to(root.resolve())
            except ValueError:
                errors.append(f"fold_runs[{index}].{key} escapes the repository")
                continue
            referenced.append(rel)
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"referenced file is missing/empty: {rel}")
                continue
            if not isinstance(expected_hash, str) or _sha256(path) != expected_hash.lower():
                errors.append(f"SHA-256 mismatch or missing digest: {rel}")
            if key == "predictions":
                try:
                    with path.open("r", encoding="utf-8-sig", newline="") as handle:
                        header = set(next(csv.reader(handle)))
                    missing = sorted(PREDICTION_COLUMNS.difference(header))
                    if missing:
                        errors.append(f"prediction schema missing {missing}: {rel}")
                except (OSError, StopIteration) as exc:
                    errors.append(f"prediction table is unreadable: {rel}: {exc}")
            elif key == "metrics":
                try:
                    metrics = json.loads(path.read_text(encoding="utf-8"))
                    required = {"seed", "fold", "macro_f1"}
                    missing = sorted(required.difference(metrics))
                    if missing:
                        errors.append(f"metrics schema missing {missing}: {rel}")
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"metrics JSON is unreadable: {rel}: {exc}")
    expected_pairs = {(seed, fold) for seed in EXPECTED_SEEDS for fold in range(5)}
    if pairs != expected_pairs:
        missing = sorted(expected_pairs.difference(pairs))
        extra = sorted(pairs.difference(expected_pairs))
        errors.append(f"fold_runs must contain exactly the 25 protocol pairs; missing={missing}, extra={extra}")
    for section in ("locked_external_validation", "censoring_audit", "figure_inputs", "edge_profiling"):
        entries = manifest.get(section)
        if not isinstance(entries, list) or not entries:
            errors.append(f"{section} must list at least one referenced artifact")
    return errors, referenced


def _matches(root: Path, patterns: tuple[str, ...]) -> list[str]:
    seen: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and path.stat().st_size > 0:
                seen.add(path.resolve())
    return [str(path.relative_to(root.resolve())) for path in sorted(seen)]


def build_report(root: Path) -> dict[str, object]:
    checks = []
    hard_blockers: list[str] = []
    for req in REQUIREMENTS:
        matches = _matches(root, req.patterns)
        passed = len(matches) >= req.minimum
        if req.hard_blocker and not passed:
            hard_blockers.append(req.key)
        checks.append(
            {
                "key": req.key,
                "label": req.label,
                "minimum": req.minimum,
                "found": len(matches),
                "passed": passed,
                "hard_blocker": req.hard_blocker,
                "matches": matches[:50],
            }
        )
    manifest_errors, manifest_files = validate_audit_manifest(root)
    manifest_passed = not manifest_errors
    if not manifest_passed:
        hard_blockers.append("audit_manifest_schema")
    checks.append(
        {
            "key": "audit_manifest_schema",
            "label": "Controlled audit manifest, schemas, and hashes",
            "minimum": 1,
            "found": 1 if manifest_passed else 0,
            "passed": manifest_passed,
            "hard_blocker": True,
            "matches": manifest_files[:50],
            "errors": manifest_errors[:100],
        }
    )
    return {
        "root": str(root.resolve()),
        "ready_for_acceptance_rereview": not hard_blockers,
        "hard_blockers": hard_blockers,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check reviewer-requested acceptance artifacts")
    parser.add_argument("--root", default=".", help="P-HLPL-HCC repository root")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)

    report = build_report(Path(args.root))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Root: {report['root']}")
        print(f"Ready for acceptance re-review: {report['ready_for_acceptance_rereview']}")
        for check in report["checks"]:
            status = "PASS" if check["passed"] else "MISSING"
            blocker = "hard" if check["hard_blocker"] else "soft"
            print(f"[{status}] {check['label']} ({check['found']}/{check['minimum']}, {blocker})")
            for match in check["matches"][:5]:
                print(f"  - {match}")
            for error in check.get("errors", [])[:10]:
                print(f"  ! {error}")
        if report["hard_blockers"]:
            print("\nHard blockers:")
            for key in report["hard_blockers"]:
                print(f"  - {key}")
    return 0 if report["ready_for_acceptance_rereview"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
