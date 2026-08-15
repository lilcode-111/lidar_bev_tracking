import argparse
import json
import subprocess
from pathlib import Path

from bev_tracking.intensity_diagnostic import load_intensity_config
from bev_tracking.v15_4_authorization import validate_formal_run_authorization
from bev_tracking.v15_4_formal import build_formal_run_plan
from bev_tracking.v15_4_materialization import load_json


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="Prepare the frozen 15.4 formal experiment plan")
    parser.add_argument("--authorization", default="outputs/intensity_filter_ablation/pre_run/formal_run_authorization.json")
    parser.add_argument("--base-config", default="configs/experiments/v15/i0_intensity_038.yaml")
    parser.add_argument("--plan-only", action="store_true", required=True, help="Day1 supports plan validation only; formal execution is added after review")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    authorization = load_json(args.authorization)
    commit = git(root, "rev-parse", "HEAD")
    clean = git(root, "status", "--porcelain") == ""
    validate_formal_run_authorization(authorization, current_commit=commit, working_tree_clean=clean, repo_root=root)
    schedule = load_json(root / "configs/experiments/v15_4/threshold_schedule.json")
    plan = build_formal_run_plan(schedule, load_intensity_config(args.base_config), authorization)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
