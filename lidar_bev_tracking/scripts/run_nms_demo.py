import json
from pathlib import Path

import cv2
import numpy as np

from bev_tracking.bev import points_to_bev
from bev_tracking.detection import make_noisy_predictions
from bev_tracking.nms import nms_bev
from bev_tracking.visualization import draw_objects


if __name__ == "__main__":
    data_dir = Path("data/sample")
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    points = np.load(data_dir / "frame_000001.npz")["points"]
    with open(data_dir / "objects_000001.json", "r", encoding="utf-8") as f:
        objects = json.load(f)["objects"]

    predictions = make_noisy_predictions(objects)
    kept_predictions = nms_bev(predictions, iou_threshold=0.3)

    bev = points_to_bev(points)
    before = draw_objects(bev, predictions, show_score=True)
    after = draw_objects(bev, kept_predictions, show_score=True)

    cv2.imwrite(str(output_dir / "nms_before.png"), before)
    cv2.imwrite(str(output_dir / "nms_after.png"), after)

    print(f"predictions before nms: {len(predictions)}")
    print(f"predictions after nms: {len(kept_predictions)}")
    print("saved outputs/figures/nms_before.png")
    print("saved outputs/figures/nms_after.png")
