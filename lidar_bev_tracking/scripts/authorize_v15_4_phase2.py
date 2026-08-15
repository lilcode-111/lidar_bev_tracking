import argparse
import subprocess
from pathlib import Path

from bev_tracking.report_writer import atomic_write_json
from bev_tracking.v15_4_authorization import build_formal_run_authorization


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="Authorize frozen 15.4 Phase-2 formal execution")
    parser.add_argument("--registry", default="configs/experiments/v15_4/t0_reference_registry.json")
    parser.add_argument("--output", default="outputs/intensity_filter_ablation/pre_run/formal_run_authorization.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    commit = git(root, "rev-parse", "HEAD")
    clean = git(root, "status", "--porcelain") == ""
    authorization = build_formal_run_authorization(
        repo_root=root,
        commit=commit,
        working_tree_clean=clean,
        schedule_path="configs/experiments/v15_4/threshold_schedule.json",
        gate_path="configs/experiments/v15_4/v15_4_release_gate.json",
        identity_path="configs/experiments/v15_4/pre_run_identity.json",
        reference_registry_path=args.registry,
    )
    atomic_write_json(args.output, authorization)
    print(f"formal_run_authorized true\nformal_comparison_commit {commit}\nsaved {args.output}")


if __name__ == "__main__":
    main()
