# 15.3.2 Day 3: Upstream Oracle Ladder

Day 3 completes the read-only per-GT oracle evidence needed before root-cause
aggregation. It does not change filtering, clustering, classification, NMS, or
evaluation.

## Stage mapping

- O3: GT points after intensity filtering.
- O4: GT points after z filtering and before intensity filtering.
- O5: GT points after ROI and before z filtering.
- O6: raw LiDAR points inside the GT box.

The implementation extracts coordinates only to fit the frozen PCA oracle. It
does not create a second point-count audit. Every extracted array length must
equal `candidate_conversion.evidence.stage_point_counts`; a mismatch aborts the
diagnostic.

## Scope and joins

O3-O6 and the Day 2 fragment oracles are emitted only on C1 frame reports.
The batch result then selects only the Day 1 frozen delta-22 and P1 control keys
and joins O1-O6 without rebuilding either cohort.

The full output preserves signed changes, including regressions:

- O2 - O1
- O2 GT-clipped - O2
- O3 - O2 and O3 - O2 GT-clipped
- O4 - O3
- O5 - O4
- O6 - O5

PCA inputs with fewer than three points or rank-deficient XY geometry retain a
null IoU with an explicit status. Root-cause labels and aggregate conclusions
remain deferred to Day 4.
