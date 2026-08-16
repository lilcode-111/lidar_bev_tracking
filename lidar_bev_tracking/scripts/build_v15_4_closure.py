import argparse
import json
from pathlib import Path

from bev_tracking.v15_4_closure import build_v15_4_closure, validate_v15_4_closure
from bev_tracking.v15_4_materialization import raw_file_sha256


def main():
    parser = argparse.ArgumentParser(description="Build or validate the read-only 15.4 closure archive")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--evidence-output", default="docs/v15_4_closure_evidence.json")
    parser.add_argument("--closure-output", default="docs/v15_4_closure.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.validate_only:
        validation = validate_v15_4_closure(
            repo_root=root,
            evidence_path=args.evidence_output,
            closure_path=args.closure_output,
        )
    else:
        _, _, validation = build_v15_4_closure(
            repo_root=root,
            evidence_output=args.evidence_output,
            closure_output=args.closure_output,
        )
    output = {
        **validation,
        "v15_4_closure_evidence": {
            "path": args.evidence_output,
            "sha256": raw_file_sha256(root / args.evidence_output),
        },
        "v15_4_closure": {
            "path": args.closure_output,
            "sha256": raw_file_sha256(root / args.closure_output),
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
