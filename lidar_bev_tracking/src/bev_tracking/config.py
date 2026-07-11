from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_KITTI_EVAL_CONFIG = {
    "data": {
        "root": "data/kitti",
        "frame_id": "000000",
    },
    "detector": {
        "eps": 0.6,
        "min_points": 20,
        "oriented": False,
    },
    "nms": {
        "iou_threshold": 0.3,
    },
    "evaluation": {
        "iou_threshold": 0.25,
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

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1].strip()
            result[current_section] = {}
            continue

        if current_section is None or ":" not in line:
            continue

        key, value = line.strip().split(":", 1)
        result[current_section][key.strip()] = parse_scalar(value.strip())

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
