import json
import copy
from pathlib import Path

import numpy as np


def make_objects():
    return [
        {"id": 1, "class_name": "car", "x": 12.0, "y": 3.0, "z": 0.0, "length": 4.5, "width": 1.9, "yaw": 0.15},
        {"id": 2, "class_name": "car", "x": 24.0, "y": -5.0, "z": 0.0, "length": 4.7, "width": 2.0, "yaw": -0.25},
        {"id": 3, "class_name": "pedestrian", "x": 18.0, "y": 7.5, "z": 0.0, "length": 0.8, "width": 0.8, "yaw": 0.0},
        {"id": 4, "class_name": "cone", "x": 8.0, "y": -8.0, "z": 0.0, "length": 0.5, "width": 0.5, "yaw": 0.0},
    ]


def sample_points_for_box(obj, num_points, rng):
    local_x = rng.uniform(-obj["length"] / 2, obj["length"] / 2, num_points)
    local_y = rng.uniform(-obj["width"] / 2, obj["width"] / 2, num_points)
    local_z = rng.uniform(-0.8, 1.2, num_points)

    c, s = np.cos(obj["yaw"]), np.sin(obj["yaw"])
    x = obj["x"] + c * local_x - s * local_y
    y = obj["y"] + s * local_x + c * local_y
    intensity = rng.uniform(0.4, 1.0, num_points)
    return np.stack([x, y, local_z, intensity], axis=1)


def generate_frame(seed=7):
    rng = np.random.default_rng(seed)
    objects = make_objects()
    points = generate_points(objects, rng)
    return points, objects


def generate_points(objects, rng):
    background = np.column_stack(
        [
            rng.uniform(0, 40, 6000),
            rng.uniform(-20, 20, 6000),
            rng.uniform(-1.6, 0.3, 6000),
            rng.uniform(0.05, 0.35, 6000),
        ]
    )

    object_points = []
    for obj in objects:
        n = 900 if obj["class_name"] == "car" else 220
        object_points.append(sample_points_for_box(obj, n, rng))

    points = np.vstack([background, *object_points]).astype(np.float32)
    return points


OBJECT_VELOCITIES = {
    1: (0.70, 0.05),
    2: (-0.25, 0.12),
    3: (0.18, -0.04),
    4: (0.00, 0.00),
}


def move_objects(objects, frame_idx):
    moved = []
    for obj in objects:
        new_obj = copy.deepcopy(obj)
        vx, vy = OBJECT_VELOCITIES.get(obj["id"], (0.0, 0.0))
        new_obj["x"] += vx * frame_idx
        new_obj["y"] += vy * frame_idx
        moved.append(new_obj)
    return moved


def generate_sequence(num_frames=8, seed=11):
    base_objects = make_objects()
    frames = []

    for frame_idx in range(num_frames):
        rng = np.random.default_rng(seed + frame_idx)
        objects = move_objects(base_objects, frame_idx)
        points = generate_points(objects, rng)
        frames.append({"frame_id": f"{frame_idx:06d}", "points": points, "objects": objects})

    return frames


def save_sample(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points, objects = generate_frame()
    np.savez_compressed(output_dir / "frame_000001.npz", points=points)

    with open(output_dir / "objects_000001.json", "w", encoding="utf-8") as f:
        json.dump({"frame_id": "000001", "objects": objects}, f, ensure_ascii=False, indent=2)
