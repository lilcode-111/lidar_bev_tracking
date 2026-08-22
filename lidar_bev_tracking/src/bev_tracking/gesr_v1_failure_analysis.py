"""Read-only, bounded GESR-v1 failure analysis for T2 material-recovery GTs."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from bev_tracking.clustering_detector import split_obstacle_filter_stages_with_indices
from bev_tracking.geometry_sanity import points_in_oriented_3d_box
from bev_tracking.gesr_v1 import POINT_TERMINAL_CODES, run_gesr_v1_optimized
from bev_tracking.kitti import (
    load_kitti_labels,
    load_kitti_point_cloud,
    resolve_kitti_calib_path,
    resolve_kitti_paths,
)
from bev_tracking.kitti_calib import kitti_labels_to_lidar_boxes, load_kitti_calib
from bev_tracking.point_retention import build_pca_oracle


T2_MATERIAL_RECOVERY_GAIN = 0.10
TOP_MISSED_POINT_COUNT = 5
IOU_REPLAY_TOLERANCE = 1.0e-8
REJECTION_CODES = tuple(code for code in POINT_TERMINAL_CODES if code != "ACCEPTED")


class GESRFailureAnalysisError(ValueError):
    pass


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_t2_material_recovery_targets(gate_result):
    if gate_result.get("Gate0_Phase2", {}).get("result") != "PASS":
        raise GESRFailureAnalysisError("failure analysis requires Gate0_Phase2 PASS")
    rows = gate_result.get("Gate_A", {}).get("delta22", [])
    targets = []
    for row in rows:
        gain = row.get("T2_minus_T0")
        if gain is not None and float(gain) >= T2_MATERIAL_RECOVERY_GAIN:
            targets.append(
                {
                    "frame_id": str(row["frame_id"]).zfill(6),
                    "gt_id": str(row["gt_id"]),
                    "formal_iou": {
                        "T0": row.get("IoU_T0"),
                        "T2": row.get("IoU_T2"),
                        "GESR-v1": row.get("IoU_GESR"),
                    },
                    "formal_gain": {
                        "T2_minus_T0": gain,
                        "GESR_minus_T0": row.get("GESR_minus_T0"),
                    },
                }
            )
    if not targets:
        raise GESRFailureAnalysisError("no T2 material-recovery GTs were found")
    return targets


def gt_point_indices(points, source_indices, gt_box):
    source_indices = np.asarray(source_indices, dtype=np.int64)
    selected_points = np.asarray(points)[source_indices]
    mask = points_in_oriented_3d_box(selected_points, gt_box)
    return np.sort(source_indices[mask])


def compact_oracle(points, source_indices, gt_box):
    oracle = build_pca_oracle(np.asarray(points)[source_indices], gt_box)
    return {
        "status": oracle["status"],
        "point_count": int(oracle["num_points"]),
        "iou": oracle["iou"],
    }


def iou_gain(left, right):
    if left is None or right is None:
        return None
    return float(left) - float(right)


def require_iou_replay(name, actual, expected, frame_id, gt_id):
    if actual is None or expected is None:
        if actual is expected:
            return
        raise GESRFailureAnalysisError(
            f"{name} PCA replay validity mismatch: {frame_id}/{gt_id}"
        )
    if abs(float(actual) - float(expected)) > IOU_REPLAY_TOLERANCE:
        raise GESRFailureAnalysisError(
            f"{name} PCA replay mismatch: {frame_id}/{gt_id}: {actual} vs {expected}"
        )


def analyze_gt_failure(
    *, points, gt_box, t0_source_indices, t2_source_indices, gesr_result, target
):
    frame_id = str(target["frame_id"]).zfill(6)
    gt_id = str(target["gt_id"])
    t0_gt = gt_point_indices(points, t0_source_indices, gt_box)
    t2_gt = gt_point_indices(points, t2_source_indices, gt_box)
    if len(np.setdiff1d(t0_gt, t2_gt, assume_unique=True)):
        raise GESRFailureAnalysisError(f"T0 is not a subset of T2: {frame_id}/{gt_id}")

    t2_added = np.setdiff1d(t2_gt, t0_gt, assume_unique=True)
    decision_by_index = {
        int(decision.source_index): decision for decision in gesr_result.candidate_decisions
    }
    missing_decisions = [int(index) for index in t2_added if int(index) not in decision_by_index]
    if missing_decisions:
        raise GESRFailureAnalysisError(
            f"T2-added GT points missing GESR decisions: {frame_id}/{gt_id}"
        )

    accepted_added = np.asarray(
        [index for index in t2_added if decision_by_index[int(index)].accepted],
        dtype=np.int64,
    )
    missed_added = np.asarray(
        [index for index in t2_added if not decision_by_index[int(index)].accepted],
        dtype=np.int64,
    )
    reason_indices = {
        reason: np.asarray(
            [
                index
                for index in missed_added
                if decision_by_index[int(index)].point_terminal_decision == reason
            ],
            dtype=np.int64,
        )
        for reason in REJECTION_CODES
    }
    attributed = sum(len(indices) for indices in reason_indices.values())
    if attributed != len(missed_added):
        raise GESRFailureAnalysisError(
            f"missed-point terminal attribution incomplete: {frame_id}/{gt_id}"
        )

    reconstructed_gesr_gt = np.union1d(t0_gt, accepted_added)
    gesr_gt = gt_point_indices(
        points,
        np.asarray(gesr_result.expanded_source_indices, dtype=np.int64),
        gt_box,
    )
    runtime_only_count = int(
        len(np.setdiff1d(gesr_gt, reconstructed_gesr_gt, assume_unique=True))
    )
    reconstructed_only_count = int(
        len(np.setdiff1d(reconstructed_gesr_gt, gesr_gt, assume_unique=True))
    )
    representations = {
        "T0": compact_oracle(points, t0_gt, gt_box),
        "T2": compact_oracle(points, t2_gt, gt_box),
        "GESR-v1": compact_oracle(points, gesr_gt, gt_box),
    }
    for name in ("T0", "T2", "GESR-v1"):
        require_iou_replay(
            name,
            representations[name]["iou"],
            target["formal_iou"][name],
            frame_id,
            gt_id,
        )

    gesr_iou = representations["GESR-v1"]["iou"]
    t0_iou = representations["T0"]["iou"]
    counterfactuals = {}
    for reason, indices in reason_indices.items():
        oracle = compact_oracle(points, np.union1d(gesr_gt, indices), gt_box)
        counterfactuals[reason] = {
            "added_point_count": int(len(indices)),
            "iou": oracle["iou"],
            "gain_vs_GESR": iou_gain(oracle["iou"], gesr_iou),
            "gain_vs_T0": iou_gain(oracle["iou"], t0_iou),
        }

    all_rejected_oracle = compact_oracle(
        points, np.union1d(gesr_gt, missed_added), gt_box
    )
    require_iou_replay(
        "all-rejected/T2",
        all_rejected_oracle["iou"],
        representations["T2"]["iou"],
        frame_id,
        gt_id,
    )

    marginal_points = []
    for source_index in missed_added:
        oracle = compact_oracle(
            points,
            np.union1d(gesr_gt, np.asarray([source_index], dtype=np.int64)),
            gt_box,
        )
        decision = decision_by_index[int(source_index)]
        marginal_points.append(
            {
                "source_index": int(source_index),
                "terminal_reason": decision.point_terminal_decision,
                "intensity": float(points[int(source_index), 3]),
                "single_point_iou_gain_vs_GESR": iou_gain(oracle["iou"], gesr_iou),
            }
        )
    marginal_points.sort(
        key=lambda item: (
            -(
                item["single_point_iou_gain_vs_GESR"]
                if item["single_point_iou_gain_vs_GESR"] is not None
                else float("-inf")
            ),
            item["source_index"],
        )
    )

    return {
        "frame_id": frame_id,
        "gt_id": gt_id,
        "formal_gain": target["formal_gain"],
        "point_counts": {
            "T0_GT": int(len(t0_gt)),
            "T2_GT": int(len(t2_gt)),
            "T2_added_GT": int(len(t2_added)),
            "GESR_accepted_from_T2_added": int(len(accepted_added)),
            "GESR_missed_from_T2_added": int(len(missed_added)),
        },
        "acceptance_rate_of_T2_added": (
            float(len(accepted_added) / len(t2_added)) if len(t2_added) else None
        ),
        "representation_identity_check": {
            "runtime_expanded_matches_T0_union_accepted_GT": (
                runtime_only_count == 0 and reconstructed_only_count == 0
            ),
            "runtime_only_point_count": runtime_only_count,
            "reconstructed_only_point_count": reconstructed_only_count,
        },
        "missed_terminal_reason_counts": {
            reason: int(len(reason_indices[reason])) for reason in REJECTION_CODES
        },
        "representations": representations,
        "counterfactual_add_rejected_reason_to_GESR": counterfactuals,
        "top_missed_single_point_marginals": marginal_points[:TOP_MISSED_POINT_COUNT],
    }


def analyze_frame(data_root, frame_id, targets):
    velodyne_path, label_path = resolve_kitti_paths(data_root, frame_id)
    points = load_kitti_point_cloud(velodyne_path)
    labels = load_kitti_labels(label_path)
    calib = load_kitti_calib(resolve_kitti_calib_path(data_root, frame_id))
    gt_by_id = {
        str(box["id"]): box for box in kitti_labels_to_lidar_boxes(labels, calib)
    }
    missing_gt = [target["gt_id"] for target in targets if target["gt_id"] not in gt_by_id]
    if missing_gt:
        raise GESRFailureAnalysisError(f"GT identity missing in {frame_id}: {missing_gt}")

    t0_stages, t0_indices = split_obstacle_filter_stages_with_indices(
        points, z_min=-0.9, intensity_min=0.38
    )
    _, t2_indices = split_obstacle_filter_stages_with_indices(
        points, z_min=-0.9, intensity_min=0.15
    )
    gesr_result = run_gesr_v1_optimized(
        frame_id,
        t0_stages["z_filter"],
        t0_indices["z_filter"],
        reason_attribution=True,
        compact_evidence=True,
    )
    return [
        analyze_gt_failure(
            points=points,
            gt_box=gt_by_id[target["gt_id"]],
            t0_source_indices=t0_indices["intensity_filter"],
            t2_source_indices=t2_indices["intensity_filter"],
            gesr_result=gesr_result,
            target=target,
        )
        for target in targets
    ]


def summarize_records(records):
    reason_counts = Counter()
    counterfactual_material_by_reason = Counter()
    totals = Counter()
    for record in records:
        totals.update(record["point_counts"])
        reason_counts.update(record["missed_terminal_reason_counts"])
        for reason, counterfactual in record[
            "counterfactual_add_rejected_reason_to_GESR"
        ].items():
            gain = counterfactual["gain_vs_T0"]
            if gain is not None and gain >= T2_MATERIAL_RECOVERY_GAIN:
                counterfactual_material_by_reason[reason] += 1
    missed_total = totals["GESR_missed_from_T2_added"]
    accepted_total = totals["GESR_accepted_from_T2_added"]
    dominant_reason = (
        min(reason_counts, key=lambda reason: (-reason_counts[reason], reason))
        if reason_counts
        else None
    )
    return {
        "target_gt_count": len(records),
        "unique_frame_count": len({record["frame_id"] for record in records}),
        "point_counts": dict(totals),
        "overall_acceptance_rate_of_T2_added": (
            float(accepted_total / (accepted_total + missed_total))
            if accepted_total + missed_total
            else None
        ),
        "missed_terminal_reason_counts": {
            reason: int(reason_counts[reason]) for reason in REJECTION_CODES
        },
        "dominant_missed_terminal_reason": dominant_reason,
        "GTs_reaching_material_recovery_when_reason_bucket_is_added_to_GESR": {
            reason: int(counterfactual_material_by_reason[reason])
            for reason in REJECTION_CODES
        },
    }


def run_failure_analysis(data_root, gate_result_path, progress_callback=None):
    gate_result = load_json(gate_result_path)
    targets = select_t2_material_recovery_targets(gate_result)
    targets_by_frame = {}
    for target in targets:
        targets_by_frame.setdefault(target["frame_id"], []).append(target)

    records = []
    ordered_frames = sorted(targets_by_frame)
    for index, frame_id in enumerate(ordered_frames, start=1):
        if progress_callback is not None:
            progress_callback(index, len(ordered_frames), frame_id, targets_by_frame[frame_id])
        records.extend(analyze_frame(data_root, frame_id, targets_by_frame[frame_id]))

    return {
        "schema_version": "15.5-gesr-v1-failure-mechanism-analysis-v1",
        "analysis_scope": "read_only_T2_material_recovery_GT",
        "selection_rule": "T2_minus_T0 >= +0.10 on frozen delta-22",
        "source_phase2_execution_commit": gate_result.get("actual_commit"),
        "phase2_conclusion_changed": False,
        "formal_result": False,
        "algorithm_modified": False,
        "parameter_search_performed": False,
        "full_point_evidence_persisted": False,
        "top_missed_point_output_bound_per_GT": TOP_MISSED_POINT_COUNT,
        "summary": summarize_records(records),
        "records": records,
    }
