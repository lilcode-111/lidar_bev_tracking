import copy

from bev_tracking.geometry import bev_iou


class Track:
    def __init__(self, detection, track_id):
        self.track_id = track_id
        self.box = copy.deepcopy(detection)
        self.class_name = detection["class_name"]
        self.vx = 0.0
        self.vy = 0.0
        self.age = 1
        self.hits = 1
        self.missed = 0
        self.history = [(self.box["x"], self.box["y"])]

    def predict(self):
        predicted = copy.deepcopy(self.box)
        predicted["x"] += self.vx
        predicted["y"] += self.vy
        return predicted

    def update(self, detection):
        prev_x, prev_y = self.box["x"], self.box["y"]
        self.box = copy.deepcopy(detection)
        self.vx = self.box["x"] - prev_x
        self.vy = self.box["y"] - prev_y
        self.age += 1
        self.hits += 1
        self.missed = 0
        self.history.append((self.box["x"], self.box["y"]))

    def mark_missed(self):
        self.box = self.predict()
        self.age += 1
        self.missed += 1
        self.history.append((self.box["x"], self.box["y"]))

    def to_box(self):
        box = copy.deepcopy(self.box)
        box["id"] = f"T{self.track_id}"
        box["track_id"] = self.track_id
        box["class_name"] = self.class_name
        return box


def match_tracks_to_detections(tracks, detections, iou_threshold):
    candidates = []
    for track_idx, track in enumerate(tracks):
        predicted_box = track.predict()
        for det_idx, det in enumerate(detections):
            if track.class_name != det["class_name"]:
                continue
            iou = bev_iou(predicted_box, det)
            if iou >= iou_threshold:
                candidates.append((iou, track_idx, det_idx))

    candidates.sort(reverse=True, key=lambda item: item[0])
    matched_tracks = set()
    matched_detections = set()
    matches = []

    for iou, track_idx, det_idx in candidates:
        if track_idx in matched_tracks or det_idx in matched_detections:
            continue
        matches.append((track_idx, det_idx, iou))
        matched_tracks.add(track_idx)
        matched_detections.add(det_idx)

    unmatched_tracks = [idx for idx in range(len(tracks)) if idx not in matched_tracks]
    unmatched_detections = [idx for idx in range(len(detections)) if idx not in matched_detections]
    return matches, unmatched_tracks, unmatched_detections


class MultiObjectTracker:
    def __init__(self, iou_threshold=0.2, max_missed=2):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks = []
        self.next_track_id = 1

    def update(self, detections):
        matches, unmatched_tracks, unmatched_detections = match_tracks_to_detections(
            self.tracks, detections, self.iou_threshold
        )

        for track_idx, det_idx, _ in matches:
            self.tracks[track_idx].update(detections[det_idx])

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for det_idx in unmatched_detections:
            self.tracks.append(Track(detections[det_idx], self.next_track_id))
            self.next_track_id += 1

        self.tracks = [track for track in self.tracks if track.missed <= self.max_missed]
        return self.tracks
