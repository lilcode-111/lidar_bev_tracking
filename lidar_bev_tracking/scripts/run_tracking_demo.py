from pathlib import Path

import cv2

from bev_tracking.bev import points_to_bev
from bev_tracking.detection import make_noisy_predictions
from bev_tracking.nms import nms_bev
from bev_tracking.synthetic import generate_sequence
from bev_tracking.tracker import MultiObjectTracker
from bev_tracking.visualization import draw_tracks


if __name__ == "__main__":
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = generate_sequence(num_frames=8)
    tracker = MultiObjectTracker(iou_threshold=0.2, max_missed=2)

    for frame_idx, frame in enumerate(frames):
        predictions = make_noisy_predictions(frame["objects"])
        detections = nms_bev(predictions, iou_threshold=0.3)
        tracks = tracker.update(detections)

        bev = points_to_bev(frame["points"])
        image = draw_tracks(bev, tracks)
        output_path = output_dir / f"tracking_frame_{frame_idx:06d}.png"
        cv2.imwrite(str(output_path), image)

        active_ids = [track.track_id for track in tracks]
        print(
            f'frame={frame["frame_id"]} detections={len(detections)} '
            f"active_tracks={active_ids} saved={output_path}"
        )
