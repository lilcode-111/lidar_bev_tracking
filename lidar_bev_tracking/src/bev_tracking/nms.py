from bev_tracking.geometry import bev_iou


def nms_bev(boxes, iou_threshold=0.3):
    sorted_boxes = sorted(boxes, key=lambda box: box.get("score", 0.0), reverse=True)
    keep = []

    while sorted_boxes:
        current = sorted_boxes.pop(0)
        keep.append(current)

        remaining = []
        for box in sorted_boxes:
            same_class = box["class_name"] == current["class_name"]
            if same_class and bev_iou(current, box) > iou_threshold:
                continue
            remaining.append(box)

        sorted_boxes = remaining

    return keep
