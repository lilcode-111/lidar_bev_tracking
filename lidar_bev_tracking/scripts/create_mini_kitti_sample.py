import argparse
from pathlib import Path

from bev_tracking.synthetic import generate_frame


def write_label_file(label_path, objects):
    lines = []
    for obj in objects:
        class_name = {
            "car": "Car",
            "pedestrian": "Pedestrian",
            "cone": "DontCare",
        }.get(obj["class_name"], "DontCare")

        if class_name == "DontCare":
            lines.append("DontCare -1 -1 -10 0 0 50 50 -1 -1 -1 -1000 -1000 -1000 -10")
            continue

        # KITTI label_2 is camera-coordinate based. This mini sample is only a
        # smoke test for file layout and parser behavior, not a calibrated GT.
        height = 1.5 if class_name == "Car" else 1.7
        width = obj["width"]
        length = obj["length"]
        x_cam = obj["y"]
        y_cam = obj["z"]
        z_cam = obj["x"]
        rotation_y = obj["yaw"]

        lines.append(
            f"{class_name} 0.00 0 0.00 0 0 50 50 "
            f"{height:.2f} {width:.2f} {length:.2f} "
            f"{x_cam:.2f} {y_cam:.2f} {z_cam:.2f} {rotation_y:.4f}"
        )

    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_calib_file(calib_path):
    # Mini sample convention: camera coordinates are [y_lidar, z_lidar, x_lidar].
    calib_text = """P0: 1 0 0 0 0 1 0 0 0 0 1 0
P1: 1 0 0 0 0 1 0 0 0 0 1 0
P2: 1 0 0 0 0 1 0 0 0 0 1 0
P3: 1 0 0 0 0 1 0 0 0 0 1 0
R0_rect: 1 0 0 0 1 0 0 0 1
Tr_velo_to_cam: 0 1 0 0 0 0 1 0 1 0 0 0
"""
    calib_path.write_text(calib_text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a tiny KITTI-layout smoke-test sample.")
    parser.add_argument("--data-root", default="data/kitti", help="Output KITTI root.")
    parser.add_argument("--frame-id", default="000000", help="Frame id to write.")
    parser.add_argument("--seed", type=int, default=7, help="Synthetic sample seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frame_id = str(args.frame_id).zfill(6)
    data_root = Path(args.data_root)
    velodyne_dir = data_root / "training" / "velodyne"
    label_dir = data_root / "training" / "label_2"
    calib_dir = data_root / "training" / "calib"
    velodyne_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    calib_dir.mkdir(parents=True, exist_ok=True)

    points, objects = generate_frame(seed=args.seed)
    bin_path = velodyne_dir / f"{frame_id}.bin"
    label_path = label_dir / f"{frame_id}.txt"
    calib_path = calib_dir / f"{frame_id}.txt"

    points.astype("float32").tofile(bin_path)
    write_label_file(label_path, objects)
    write_calib_file(calib_path)

    print(f"saved mini KITTI point cloud: {bin_path}")
    print(f"saved mini KITTI labels: {label_path}")
    print(f"saved mini KITTI calib: {calib_path}")
    print("note: this is a synthetic smoke-test sample, not a real KITTI frame")
