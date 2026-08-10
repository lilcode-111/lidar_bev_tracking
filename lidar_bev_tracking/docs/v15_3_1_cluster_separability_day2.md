# v15.3.1 Cluster Separability Diagnostic — Day 2

Day 2 adds read-only measurements to every Day 1 P1/P2/N cluster record:

- point count, axis length/width, height span, XY density;
- original PCA-box length/width;
- target-GT coverage and cluster purity for P1/P2;
- original PCA-box versus target-GT oracle IoU for P2 only.

The target GT is selected deterministically by maximum in-GT cluster-point count,
then GT id. Per-association counts and ratios remain in the record. N retains the
same geometry fields but has null GT and oracle fields.

Oracle evaluation does not change class labels, feed P2 into NMS or Evaluation,
or alter any detector output. Day 3 will compare distributions by frame and
distance bin and produce the final diagnosis.

The batch path builds the Day 1 grouped result exactly once. Day 2 accepts that
validated result directly; it does not re-index frame reports, recompute the
C0-to-C1 delta cohort, or repeat P1/P2/N grouping.
