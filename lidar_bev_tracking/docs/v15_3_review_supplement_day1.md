# v15.3 Review Supplement Day 1

## C1 delta cohort

The supplement compares canonical `candidate_conversion.evidence` for C0 and
C1 using the unique `(frame_id, gt_id)` key. It selects GTs associated with a
cluster in C1 but not in C0, then reports their C1 terminal states and compact
downstream conversion fields.

The original GT universe is not changed. This is a cohort difference over the
same fixed positive-GT set.

The ablation output now contains a top-level `delta_cohort` with:

- base and candidate associated-GT counts;
- `new_associated_gt_count`;
- terminal-state counts;
- classification/NMS/IoU stage counts;
- one compact record per newly associated GT.
