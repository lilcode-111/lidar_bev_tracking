import argparse
import json
from pathlib import Path

from bev_tracking.experiment_gate import load_a0_prime_declaration
from bev_tracking.failure_analysis import load_batch_result_from_run_directory
from bev_tracking.failure_evidence_batch import load_diagnostic_manifest
from bev_tracking.intensity_diagnostic import (
    build_intensity_comparison,
    load_intensity_config,
    validate_i0_reproduces_baseline,
    validate_intensity_config_pair,
    validate_manifest_against_declaration,
    validate_report_invariants,
)
from bev_tracking.report_writer import atomic_write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Build Day4 analysis from existing I0/I1 diagnostic reports.")
    parser.add_argument("--i0-report", default="outputs/intensity_diagnostic/i0_intensity_038.json")
    parser.add_argument("--i1-report", default="outputs/intensity_diagnostic/i1_intensity_000.json")
    parser.add_argument("--i0-config", default="configs/experiments/v15/i0_intensity_038.yaml")
    parser.add_argument("--i1-config", default="configs/experiments/v15/i1_intensity_000.yaml")
    parser.add_argument(
        "--baseline-declaration",
        default="configs/experiments/v15/a0_prime_geometry_corrected.json",
    )
    parser.add_argument("--output", default="outputs/intensity_diagnostic/i0_i1_comparison.json")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    declaration = load_a0_prime_declaration(args.baseline_declaration, repo_root=repo_root)
    i0_config = load_intensity_config(args.i0_config)
    i1_config = load_intensity_config(args.i1_config)
    config_diff = validate_intensity_config_pair(i0_config, i1_config)
    manifest = load_diagnostic_manifest(i0_config["data"]["manifest"])
    manifest_gate = validate_manifest_against_declaration(manifest, declaration)

    i0_report = read_json(args.i0_report)
    i1_report = read_json(args.i1_report)
    i0_invariants = validate_report_invariants(i0_report, i0_config, manifest)
    i1_invariants = validate_report_invariants(
        i1_report,
        i1_config,
        manifest,
        require_intensity_identity=True,
    )
    baseline_batch = load_batch_result_from_run_directory(repo_root / declaration["source_run"]["run_directory"])
    i0_reproduction = validate_i0_reproduces_baseline(i0_report, baseline_batch, manifest)

    comparison = build_intensity_comparison(
        declaration,
        manifest_gate,
        config_diff,
        i0_report,
        i1_report,
        i0_invariants,
        i1_invariants,
        i0_reproduction,
    )
    output_path = Path(args.output)
    atomic_write_json(output_path, comparison)
    print("existing I0/I1 reports passed all gates")
    print(f"saved {output_path}")
    return 0


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"diagnostic report not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"diagnostic report must be a JSON object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
