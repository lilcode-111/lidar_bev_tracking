import json
from pathlib import Path

import cv2
import numpy as np

from bev_tracking.bev import points_to_bev
from bev_tracking.visualization import draw_objects


if __name__ == "__main__":
    data_dir = Path("data/sample")
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    points = np.load(data_dir / "frame_000001.npz")["points"]
    with open(data_dir / "objects_000001.json", "r", encoding="utf-8") as f:
        objects = json.load(f)["objects"]

    bev = points_to_bev(points)
    image = draw_objects(bev, objects)

    output_path = output_dir / "bev_frame_000001.png"
    cv2.imwrite(str(output_path), image)
    print(f"saved {output_path}")
