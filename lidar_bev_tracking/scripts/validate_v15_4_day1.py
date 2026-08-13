import argparse
import json

from bev_tracking.v15_4_materialization import validate_day1_materialization


def main():
    parser = argparse.ArgumentParser(description="Validate 15.4 Phase-1 Day1 frozen material")
    parser.add_argument("--schedule", default="configs/experiments/v15_4/threshold_schedule.json")
    parser.add_argument("--identity", default="configs/experiments/v15_4/pre_run_identity.json")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    result = validate_day1_materialization(args.schedule, args.identity, repo_root=args.repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
