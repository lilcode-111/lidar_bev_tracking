# 15.3.2 Day 4: Root-Cause Aggregation

Day 4 closes the implementation of the read-only Car Point Retention & Fragment
Recoverability Diagnostic. It consumes the frozen Day 1 cohorts and the joined
Day 3 ladder; it does not replay, refit, or alter pipeline evidence itself.

## Per-GT attribution

A stage is material when its preregistered signed IoU gain is at least 0.10:

- O2 - O1: `CLUSTER_FRAGMENTATION_LIMITED`.
- O2 GT-clipped - O2 or O3 - O2 GT-clipped: `CLUSTER_FORMATION_LIMITED`.
- O4 - O3: `INTENSITY_FILTER_LIMITED`.
- O5 - O4: `Z_FILTER_LIMITED`.
- O6 - O5: `ROI_FILTER_LIMITED`.

Two or more distinct material stage labels produce `MIXED`. When there is no
material stage signal, an invalid O6 or O6 IoU below 0.25 produces
`RAW_GEOMETRY_OBSERVABILITY_LIMITED`; a recoverable O6 with no material step is
`UNRESOLVED`. A single material stage signal remains the primary attribution
even when raw geometry is also weak, while `raw_recoverable` preserves that
secondary fact.

The GT-clipped O2 control is used for cluster-formation attribution so that
missing vehicle points are not confused with background contamination.

## Final report

The `point_retention_day4` output contains annotated delta-22 and P1-control
records plus:

- root-cause counts, dominant label, ratio, and majority status;
- O1-O6 valid/null, median/mean/range, and IoU 0.25/0.50 counts;
- every signed-delta distribution and material-gain count;
- raw/ROI/z/intensity and associated-cluster point-count distributions;
- root-cause counts by distance bin.

The real 25-frame replay is performed only after unit and regression tests pass.
