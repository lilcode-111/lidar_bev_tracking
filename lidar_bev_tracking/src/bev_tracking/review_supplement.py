"""Compact, read-only evidence summaries for the v15.3 review supplement."""

from collections import Counter

from bev_tracking.result_types import CandidateConversionState


DISTANCE_BINS = ("near_0_15", "mid_15_30", "far_30_inf", "total")
TERMINAL_STATES = tuple(state.value for state in CandidateConversionState)


def _records_by_variant(reports_by_variant):
    output = {}
    for variant, reports in reports_by_variant.items():
        records = []
        seen = set()
        for report in reports:
            conversion = report.get("candidate_conversion") or {}
            evidence = conversion.get("evidence")
            if not isinstance(evidence, list):
                raise ValueError(f"missing canonical evidence for {variant}: {report.get('frame_id')}")
            for item in evidence:
                key = (str(item.get("frame_id")).zfill(6), str(item.get("gt_id")))
                if key in seen:
                    raise ValueError(f"duplicate canonical evidence: {variant} {key}")
                seen.add(key)
                records.append(item)
        output[variant] = records
    return output


def build_terminal_state_matrix(reports_by_variant):
    """Return complete C0/C1 terminal-state counts by distance and total."""
    records_by_variant = _records_by_variant(reports_by_variant)
    matrix = {}
    for variant, records in records_by_variant.items():
        grouped = {distance_bin: [] for distance_bin in DISTANCE_BINS}
        for record in records:
            distance_bin = record.get("distance_bin")
            if distance_bin not in DISTANCE_BINS[:-1]:
                raise ValueError(f"invalid distance bin for {variant}: {distance_bin}")
            grouped[distance_bin].append(record)
            grouped["total"].append(record)
        matrix[variant] = {}
        for distance_bin, grouped_records in grouped.items():
            counts = Counter(record.get("terminal_state", "unknown") for record in grouped_records)
            matrix[variant][distance_bin] = {
                state: int(counts.get(state, 0)) for state in TERMINAL_STATES
            }
            if counts.get("unknown", 0):
                raise ValueError(f"unknown terminal state for {variant} {distance_bin}")
    return matrix


def _mean(values):
    return float(sum(values) / len(values)) if values else None


def _geometry_summary(records):
    values = {field: [] for field in ("center_error_m", "length_error_m", "width_error_m", "yaw_error_rad")}
    candidate_count = 0
    competition_count = 0
    for record in records:
        downstream = record.get("downstream_attribution") or {}
        geometry = (downstream.get("geometry") or {}).get("candidates") or []
        candidate_count += len(geometry)
        evaluation = downstream.get("evaluation") or {}
        competition_count += int(bool(evaluation.get("competition_matches")))
        for candidate in geometry:
            for field in values:
                if field in candidate:
                    values[field].append(float(candidate[field]))
    return {
        "associated_car_candidate_count_after_nms": sum(
            bool(record.get("car_detection_ids_after_nms")) for record in records
        ),
        "geometry_candidate_count": int(candidate_count),
        "geometry_error_mean": {field: _mean(items) for field, items in values.items()},
        "evaluation_competition_gt_count": int(competition_count),
        "iou_ge_0_50_but_unmatched_count": sum(
            record.get("terminal_state") == "iou_ge_0_50_but_unmatched" for record in records
        ),
        "matched_at_0_50_count": sum(
            record.get("terminal_state") == "matched_at_0_50" for record in records
        ),
    }


def build_review_supplement_day2(reports_by_variant):
    """Build the review-only terminal matrix and minimal downstream summary."""
    records_by_variant = _records_by_variant(reports_by_variant)
    return {
        "schema_version": "15.3-review-1",
        "terminal_state_matrix": build_terminal_state_matrix(reports_by_variant),
        "downstream_summary": {
            variant: _geometry_summary(records)
            for variant, records in records_by_variant.items()
        },
    }
