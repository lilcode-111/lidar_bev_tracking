# v15.3.2 Car Point Retention & Fragment Recoverability — Day 1

Day 1 freezes the delta-22 and P1-control keys against canonical 15.3.1 evidence,
reuses per-GT raw/ROI/z/intensity counts from
`candidate_conversion.evidence.stage_point_counts`, and defines the O1–O6
contract. It does not recrop stage point arrays, calculate a new oracle box, or
alter any pipeline stage.

PCA inputs require at least three points and XY rank two. Other inputs are reported
as `insufficient_points` or `degenerate_geometry`; no synthetic 0.1 m box is treated
as valid evidence. Existing canonical GT stage counts must be monotonically
non-increasing. Future oracle extraction will compare extracted array lengths back
to these canonical counts as an equality gate.

The ladder includes the auxiliary `O2_gt_clipped` contamination control and the
`ROI_FILTER_LIMITED` outcome. Signed IoU deltas are preserved, and a material gain
is preregistered as an absolute IoU increase of at least 0.10.
