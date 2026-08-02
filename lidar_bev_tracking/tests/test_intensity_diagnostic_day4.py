import unittest

from bev_tracking.intensity_diagnostic import (
    IntensityDiagnosticError,
    build_distance_analysis,
    build_gt_incremental_analysis,
    build_per_frame_change_analysis,
    build_zero_detection_analysis,
)


def metrics(tp, fp, fn, effective):
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "neutralized_detections": 0,
        "effective_car_detection_count": effective,
    }


def gt_record(frame_id, gt_id, distance_bin, intensity_points, cluster, car, iou, matched_050, matched_025):
    cluster_ids = ["cluster_1"] if cluster else []
    car_ids = ["cluster_1"] if car else []
    if not cluster:
        outcome = "no_cluster"
    elif not car:
        outcome = "raw_detection_rejected_as_non_car"
    elif matched_050:
        outcome = "matched_at_primary_iou"
    else:
        outcome = "final_car_candidate_below_primary_iou"
    ranges = {"near_0_15": 10.0, "mid_15_30": 20.0, "far_30_inf": 35.0}
    return {
        "frame_id": frame_id,
        "gt_id": gt_id,
        "range_xy_m": ranges[distance_bin],
        "distance_bin": distance_bin,
        "stage_point_counts": {
            "raw": 30,
            "roi": 30,
            "z_filter": 20,
            "intensity_filter": intensity_points,
        },
        "cluster_ids": cluster_ids,
        "raw_detection_ids": cluster_ids,
        "car_detection_ids_before_nms": car_ids,
        "car_detection_ids_after_nms": car_ids,
        "best_iou_after_nms": iou,
        "matched_by_iou": {"0.50": matched_050, "0.25": matched_025},
        "candidate_outcome": outcome,
    }


def report(records, frame_metrics):
    frames = []
    for frame_id, iou_metrics in frame_metrics.items():
        frames.append(
            {
                "frame_id": frame_id,
                "failure_evidence": {
                    "summary": {
                        "metrics_by_iou": iou_metrics,
                        "primary_reason_counts": {},
                    }
                },
            }
        )
    return {
        "summary": {
            "candidate_coverage": {
                "counts": {"num_positive_gt": len(records)},
            }
        },
        "gt_candidate_records": records,
        "frames": frames,
    }


class IntensityDiagnosticDay4Test(unittest.TestCase):
    def setUp(self):
        self.i0_records = [
            gt_record("000001", "gt_1", "near_0_15", 2, False, False, 0.0, False, False),
            gt_record("000001", "gt_2", "mid_15_30", 4, True, False, 0.0, False, False),
            gt_record("000002", "gt_3", "far_30_inf", 1, False, False, 0.0, False, False),
        ]
        self.i1_records = [
            gt_record("000001", "gt_1", "near_0_15", 20, True, True, 0.6, True, True),
            gt_record("000001", "gt_2", "mid_15_30", 20, True, True, 0.3, False, True),
            gt_record("000002", "gt_3", "far_30_inf", 20, False, False, 0.0, False, False),
        ]
        i0_metrics = {
            "000001": {"0.50": metrics(0, 1, 2, 1), "0.25": metrics(0, 1, 2, 1)},
            "000002": {"0.50": metrics(0, 0, 1, 0), "0.25": metrics(0, 0, 1, 0)},
        }
        i1_metrics = {
            "000001": {"0.50": metrics(1, 2, 1, 3), "0.25": metrics(2, 1, 0, 3)},
            "000002": {"0.50": metrics(0, 1, 1, 1), "0.25": metrics(0, 1, 1, 1)},
        }
        self.i0_report = report(self.i0_records, i0_metrics)
        self.i1_report = report(self.i1_records, i1_metrics)

    def test_gt_incremental_analysis_tracks_new_candidates_and_fn_to_tp(self):
        analysis = build_gt_incremental_analysis(self.i0_report, self.i1_report)

        self.assertEqual(analysis["summary"]["new_cluster_gt_count"], 1)
        self.assertEqual(analysis["summary"]["new_car_candidate_gt_count"], 2)
        self.assertEqual(analysis["summary"]["fn_to_tp_by_iou"]["0.50"], 1)
        self.assertEqual(analysis["summary"]["fn_to_tp_by_iou"]["0.25"], 2)
        self.assertEqual(analysis["summary"]["tp_to_fn_by_iou"]["0.50"], 0)
        self.assertEqual(analysis["records"][0]["intensity_points_added"], 18)

    def test_distance_analysis_reports_near_mid_far_coverage(self):
        analysis = build_distance_analysis(self.i0_report, self.i1_report)

        self.assertEqual(analysis["near_0_15"]["i1"]["gt_with_cluster"], 1)
        self.assertEqual(analysis["mid_15_30"]["i1"]["metrics_by_iou"]["0.25"]["tp"], 1)
        self.assertEqual(analysis["far_30_inf"]["i1"]["zero_car_candidate_gt"], 1)

    def test_per_frame_analysis_counts_changes_and_uses_stable_top_order(self):
        analysis = build_per_frame_change_analysis(self.i0_report, self.i1_report)

        distribution = analysis["change_distributions_by_iou"]["0.50"]
        self.assertEqual(distribution["tp"]["increased_frames"], 1)
        self.assertEqual(distribution["tp"]["unchanged_frames"], 1)
        top_fp = analysis["top_changes_by_iou"]["0.50"]["fp_growth_top5"]
        self.assertEqual([item["frame_id"] for item in top_fp], ["000001", "000002"])

    def test_gt_join_rejects_changed_frozen_stage_counts(self):
        self.i1_records[0]["stage_point_counts"]["z_filter"] = 19

        with self.assertRaises(IntensityDiagnosticError):
            build_gt_incremental_analysis(self.i0_report, self.i1_report)

    def test_zero_detection_analysis_reports_drop_and_breakdown(self):
        i0 = invariant_coverage(zero_count=10, no_cluster=6, rejected=4)
        i1 = invariant_coverage(zero_count=4, no_cluster=1, rejected=3)

        analysis = build_zero_detection_analysis(i0, i1)

        self.assertEqual(analysis["decrease_count"], 6)
        self.assertAlmostEqual(analysis["decrease_ratio"], 0.6)
        self.assertEqual(analysis["zero_candidate_breakdown"]["no_cluster"]["delta"], -5)


def invariant_coverage(zero_count, no_cluster, rejected):
    return {
        "candidate_coverage": {
            "counts": {"num_positive_gt": 12},
            "zero_car_candidate_gt_count": zero_count,
            "candidate_outcome_counts": {
                "no_cluster": no_cluster,
                "raw_detection_rejected_as_non_car": rejected,
            },
        }
    }


if __name__ == "__main__":
    unittest.main()
