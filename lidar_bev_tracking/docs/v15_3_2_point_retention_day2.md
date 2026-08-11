# v15.3.2 Car Point Retention & Fragment Recoverability — Day 2

Day 2 implements the fragmentation section of the frozen ladder:

- `O1`: the valid single associated cluster with the highest canonical PCA IoU;
- `O2`: frozen PCA over the exact-row unique union of all canonical associated
  cluster points;
- `O2_gt_clipped`: the same union restricted to the target GT box, used only to
  expose contamination effects.

Canonical `candidate_conversion.evidence.cluster_ids` define association. No
association or clustering is recalculated. Every valid O1 IoU must equal its
15.3.1 `candidate_branches[].candidate_iou` value within `1e-9`.

Invalid PCA inputs produce a status and null IoU. O2 records input, unique-union,
duplicate, and GT-clipped point counts plus signed `O2-O1` and
`O2_gt_clipped-O2` changes. Nothing is sent to classifier, NMS, or Evaluation.
