import argparse
import sys

from bev_tracking.failure_analysis import FailureAnalysisError, generate_failure_cases_from_run_directory
from bev_tracking.report_writer import ReportWriteError, write_failure_cases_report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate ranked failure cases from a completed batch run directory.")
    parser.add_argument("--run-dir", required=True, help="Path to a complete Batch Robustness run directory.")
    parser.add_argument("--top-k", type=int, default=5, help="Maximum cases per failure category.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        failure_cases = generate_failure_cases_from_run_directory(args.run_dir, top_k=args.top_k)
        payload, output_path = write_failure_cases_report(args.run_dir, failure_cases, top_k=args.top_k)
    except (FailureAnalysisError, ReportWriteError, ValueError) as exc:
        print(f"failure analysis failed: {exc}", file=sys.stderr)
        return 1

    print(f"source run: {payload['source']['run_id']}")
    print(f"failure cases: {payload['summary']['total_failure_cases']}")
    for category, count in payload["summary"]["counts_by_category"].items():
        print(f"{category}: {count}")
    print(f"saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
