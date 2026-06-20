import numpy as np


def box_corners_bev(box):
    length, width = box["length"], box["width"]
    corners = np.array(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ],
        dtype=np.float32,
    )

    c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return corners @ rot.T + np.array([box["x"], box["y"]], dtype=np.float32)


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    pts = np.asarray(poly, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _is_inside(point, edge_start, edge_end):
    edge = edge_end - edge_start
    to_point = point - edge_start
    return edge[0] * to_point[1] - edge[1] * to_point[0] <= 1e-6


def _line_intersection(p1, p2, q1, q2):
    r = p2 - p1
    s = q2 - q1
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-8:
        return p2
    t = ((q1 - p1)[0] * s[1] - (q1 - p1)[1] * s[0]) / denom
    return p1 + t * r


def polygon_clip(subject_polygon, clip_polygon):
    output = [np.asarray(p, dtype=np.float32) for p in subject_polygon]
    clip = [np.asarray(p, dtype=np.float32) for p in clip_polygon]

    for i in range(len(clip)):
        edge_start = clip[i]
        edge_end = clip[(i + 1) % len(clip)]
        input_list = output
        output = []

        if not input_list:
            break

        prev = input_list[-1]
        for curr in input_list:
            curr_inside = _is_inside(curr, edge_start, edge_end)
            prev_inside = _is_inside(prev, edge_start, edge_end)

            if curr_inside:
                if not prev_inside:
                    output.append(_line_intersection(prev, curr, edge_start, edge_end))
                output.append(curr)
            elif prev_inside:
                output.append(_line_intersection(prev, curr, edge_start, edge_end))

            prev = curr

    return output


def bev_iou(box_a, box_b):
    poly_a = box_corners_bev(box_a)
    poly_b = box_corners_bev(box_b)

    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)
    inter_poly = polygon_clip(poly_a, poly_b)
    inter_area = polygon_area(inter_poly)
    union = area_a + area_b - inter_area

    if union <= 1e-8:
        return 0.0
    return inter_area / union
