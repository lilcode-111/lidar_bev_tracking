import copy


def make_noisy_predictions(objects):
    predictions = []

    for obj in objects:
        base = copy.deepcopy(obj)
        base["score"] = 0.92 if obj["class_name"] == "car" else 0.84
        predictions.append(base)

        duplicate = copy.deepcopy(obj)
        duplicate["id"] = f'{obj["id"]}_dup'
        duplicate["x"] += 0.25
        duplicate["y"] -= 0.18
        duplicate["yaw"] += 0.04
        duplicate["score"] = 0.68
        predictions.append(duplicate)

    predictions.append(
        {
            "id": "fp_1",
            "class_name": "car",
            "x": 31.0,
            "y": 12.0,
            "z": 0.0,
            "length": 4.4,
            "width": 1.9,
            "yaw": -0.1,
            "score": 0.42,
        }
    )

    return predictions
