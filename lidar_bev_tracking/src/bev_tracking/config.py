from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_KITTI_EVAL_CONFIG = {
    "data": {
        "root": "data/kitti",
        "frame_id": "000000",
        "frame_ids": ["000000"],
    },
    "detector": {
        "eps": 0.6,
        "min_points": 20,
        "oriented": False,
        "z_min": -0.9,
        "intensity_min": 0.38,
    },
    "nms": {
        "iou_threshold": 0.3,
    },
    "evaluation": {
        "iou_threshold": 0.5,
        "auxiliary_iou_thresholds": [0.25],
    },
    "outputs": {
        "report_dir": "outputs/reports",
    },
}


def load_yaml_config(config_path):
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        text = f.read()
    loaded = yaml.safe_load(text) if yaml is not None else parse_simple_yaml(text)
    loaded = loaded or {}
    return merge_dicts(DEFAULT_KITTI_EVAL_CONFIG, loaded)


def merge_dicts(base, override):
    merged = {}
    for key, value in base.items():
        if isinstance(value, dict):
            merged[key] = merge_dicts(value, override.get(key, {}))
        else:
            merged[key] = override.get(key, value)

    for key, value in override.items():
        if key not in merged:
            merged[key] = value

    return merged


def parse_simple_yaml(text):
    result = {}
    current_section = None
    current_list_key = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1].strip()
            result[current_section] = {}
            current_list_key = None
            continue

        if current_section is None or ":" not in line:
            if current_section is not None and current_list_key is not None and line.strip().startswith("- "):
                result[current_section][current_list_key].append(parse_scalar(line.strip()[2:].strip()))
            continue

        key, value = line.strip().split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            result[current_section][key] = parse_scalar(value)
            current_list_key = None
        else:
            result[current_section][key] = []
            current_list_key = key

    return result


def parse_scalar(value):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
