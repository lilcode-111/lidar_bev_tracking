# 15.3.2 O2 Point Identity Fix — Closure Package

The closure command reads the existing 15.3.2 result as immutable before
evidence and replays only frozen C1 filtering/clustering to recover source point
indices. It does not call classifier, NMS, Evaluation, or metric aggregation.

The replay is guarded by source-index reconstruction, cluster point count, and
per-fragment O1 IoU equality. It replaces only O2, O2 GT-clipped, their dependent
signed deltas, and Day 4 attribution. O1 and O3-O6 are copied from the frozen
source result.

The output contains the requested delta-22 per-GT O2 before/after, material
signal counts, review of the original four cluster-formation-plus-intensity
samples, root-cause counts, and the invariant O4-O3 material count. A sibling
`.sha256` text file identifies the exact corrected JSON artifact.
