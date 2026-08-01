import argparse
from pathlib import Path

from bev_tracking.experiment_gate import load_a0_prime_declaration
from bev_tracking.failure_analysis import load_batch_result_from_run_directory
from bev_tracking.failure_evidence_batch import load_diagnostic_manifest
from bev_tracking.intensity_diagnostic import (
    build_intensity_comparison,
    load_intensity_config,
    run_diagnostic_from_config,
    validate_i0_reproduces_baseline,
    validate_intensity_config_pair,
    validate_manifest_against_declaration,
    validate_report_invariants,
)
from bev_tracking.report_writer import atomic_write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run the frozen 25-frame I0/I1 intensity diagnostic experiment.")
    parser.add_argument("--i0-config", default="configs/experiments/v15/i0_intensity_038.yaml")
    parser.add_argument("--i1-config", default="configs/experiments/v15/i1_intensity_000.yaml")
    parser.add_argument(
        "--baseline-declaration",
        default="configs/experiments/v15/a0_prime_geometry_corrected.json",
    )
    parser.add_argument("--output-dir", default="outputs/intensity_diagnostic")
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
    baseline_batch = load_batch_result_from_run_directory(repo_root / declaration["source_run"]["run_directory"])

    output_dir = Path(args.output_dir)
    i0_path = output_dir / "i0_intensity_038.json"
    i1_path = output_dir / "i1_intensity_000.json"
    comparison_path = output_dir / "i0_i1_comparison.json"

    print("running I0: intensity_min=0.38")
    i0_report = run_diagnostic_from_config(
        i0_config,
        manifest,
        source_run_id=declaration["source_run"]["run_id"],
        progress_callback=progress("I0"),
    )
    atomic_write_json(i0_path, i0_report)
    i0_invariants = validate_report_invariants(i0_report, i0_config, manifest)
    i0_reproduction = validate_i0_reproduces_baseline(i0_report, baseline_batch, manifest)
    print("I0 reproduced A0 prime on all 25 frames")

    print("running I1: intensity_min=0.0")
    i1_report = run_diagnostic_from_config(
        i1_config,
        manifest,
        source_run_id=declaration["source_run"]["run_id"],
        progress_callback=progress("I1"),
    )
    atomic_write_json(i1_path, i1_report)
    i1_invariants = validate_report_invariants(
        i1_report,
        i1_config,
        manifest,
        require_intensity_identity=True,
    )

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
    atomic_write_json(comparison_path, comparison)
    print("all I0/I1 invariants passed")
    print(f"saved {i0_path}")
    print(f"saved {i1_path}")
    print(f"saved {comparison_path}")
    return 0


def progress(label):
    return lambda index, total, frame_id: print(f"[{label} {index:02d}/{total:02d}] replay {frame_id}")


if __name__ == "__main__":
    raise SystemExit(main())
