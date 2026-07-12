POSITIVE_GT_CLASSES = {"car"}
NEUTRAL_GT_CLASSES = {"van", "truck"}
EXCLUDED_GT_CLASSES = {"pedestrian", "cyclist", "person_sitting", "tram", "misc"}
DONTCARE_CLASSES = {"dontcare"}
POSITIVE_DETECTION_CLASSES = {"car"}
ROI_X_RANGE = (0.0, 40.0)
ROI_Y_RANGE = (-20.0, 20.0)
PRIMARY_IOU_THRESHOLD = 0.5
AUXILIARY_IOU_THRESHOLDS = (0.25,)
NEUTRAL_IOU_THRESHOLD = 0.5


def normalize_class_name(class_name):
    return str(class_name).strip().lower()


def is_center_inside_roi(box, x_range=ROI_X_RANGE, y_range=ROI_Y_RANGE):
    x = box.get("x")
    y = box.get("y")
    if x is None or y is None:
        return False
    return x_range[0] <= float(x) < x_range[1] and y_range[0] <= float(y) < y_range[1]


def classify_gt_box(box):
    class_name = normalize_class_name(box.get("class_name", ""))
    if not is_center_inside_roi(box):
        return "outside_roi"
    if class_name in POSITIVE_GT_CLASSES:
        return "positive"
    if class_name in NEUTRAL_GT_CLASSES:
        return "neutral"
    if class_name in EXCLUDED_GT_CLASSES:
        return "excluded"
    if class_name in DONTCARE_CLASSES:
        return "dontcare"
    return "excluded"


def is_positive_detection(box):
    class_name = normalize_class_name(box.get("class_name", ""))
    return class_name in POSITIVE_DETECTION_CLASSES and is_center_inside_roi(box)


def assign_det_indices(detections):
    output = []
    for idx, det in enumerate(detections):
        item = dict(det)
        item.setdefault("det_index", idx)
        output.append(item)
    return output


def detection_sort_key(box):
    return (-float(box.get("score", 0.0)), int(box.get("det_index", 0)))


def safe_divide(numerator, denominator):
    if denominator == 0:
        return None
    return float(numerator / denominator)


def safe_f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return None
    return float((2 * tp) / denominator)
