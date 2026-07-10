from pathlib import Path

import cv2

from bev_tracking.bev import points_to_bev
from bev_tracking.clustering_detector import detect_objects_from_points
from bev_tracking.nms import nms_bev
from bev_tracking.synthetic import generate_frame
from bev_tracking.visualization import draw_objects


if __name__ == "__main__":
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    points, _ = generate_frame(seed=7)
    raw_detections = detect_objects_from_points(points, eps=0.6, min_points=20, oriented=True)
    detections = nms_bev(raw_detections, iou_threshold=0.3)

    bev = points_to_bev(points)
    image = draw_objects(bev, detections, show_score=True)
    output_path = output_dir / "oriented_clustering_detection.png"
    cv2.imwrite(str(output_path), image)

    print(f"raw detections: {len(raw_detections)}")
    print(f"detections after nms: {len(detections)}")
    for det in detections:
        print(
            f'{det["id"]} {det["class_name"]} '
            f'x={det["x"]:.2f} y={det["y"]:.2f} '
            f'l={det["length"]:.2f} w={det["width"]:.2f} '
            f'yaw={det["yaw"]:.2f} '
            f'points={det["num_points"]} score={det["score"]:.2f}'
        )
    print(f"saved {output_path}")
