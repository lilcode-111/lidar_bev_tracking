import json
from pathlib import Path

from bev_tracking.v15_4_materialization import load_json, raw_file_sha256, validate_pre_run_identity, validate_threshold_schedule
from bev_tracking.v15_4_release import ARTIFACT_SCHEMA_VERSION, validate_release_gate_config


def main():
    root = Path(__file__).resolve().parents[1]
    schedule_path = root / "configs/experiments/v15_4/threshold_schedule.json"
    identity_path = root / "configs/experiments/v15_4/pre_run_identity.json"
    gate_path = root / "configs/experiments/v15_4/v15_4_release_gate.json"
    output = {
        "schema_version": "15.4-phase1-day4-validation-v1",
        "schedule": validate_threshold_schedule(load_json(schedule_path)),
        "identity": validate_pre_run_identity(load_json(identity_path), repo_root=root),
        "release_gate": validate_release_gate_config(load_json(gate_path)),
        "artifacts": {
            "threshold_schedule": {"path": schedule_path.relative_to(root).as_posix(), "sha256": raw_file_sha256(schedule_path)},
            "release_gate": {"path": gate_path.relative_to(root).as_posix(), "sha256": raw_file_sha256(gate_path)},
            "formal_artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        },
        "phase_1_day4_code_complete": True,
        "formal_results_observed": False,
        "formal_run_authorized": False,
        "next_required_step": "Commit the clean materialization tree, freeze that commit, then create the post-commit authorization record before Phase-2.",
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
