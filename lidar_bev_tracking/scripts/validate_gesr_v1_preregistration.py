import json
from pathlib import Path

from bev_tracking.gesr_v1_preregistration import validate_preregistration_artifacts


def main():
    root = Path(__file__).resolve().parents[1]
    result = validate_preregistration_artifacts(root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
