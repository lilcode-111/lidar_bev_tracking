import argparse
import json
from pathlib import Path

from bev_tracking.v15_5_seed_support import build_phase0_day1


def main():
    parser = argparse.ArgumentParser(description="Build read-only 15.5 Phase-0 Day1 seed-support identities")
    parser.add_argument("--output", default="outputs/seed_support_selectivity/v15_5_phase0_day1.json")
    parser.add_argument("--cache-dir", default="outputs/seed_support_selectivity/day1_cache")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = build_phase0_day1(
        repo_root=root,
        output_path=args.output,
        cache_dir=args.cache_dir,
    )
    summary = {
        "schema_version": result["schema_version"],
        "day1_complete": result["day1_complete"],
        "identity_validation": result["identity_validation"],
        "band_totals": result["band_totals"],
        "gt_oracle_leakage": result["contracts"]["gt_oracle_leakage"],
        "formal_pipeline_rerun": result["formal_pipeline_rerun"],
        "saved": args.output,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
