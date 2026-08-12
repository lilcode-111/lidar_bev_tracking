# 15.3.2 O2 Point Identity Fix — Day 1

This narrow audit fix replaces coordinate-row identity with original raw-LiDAR
row identity for O2 and O2 GT-clipped.

## Frozen semantics

The raw point array receives deterministic per-frame indices `0..N-1`. ROI,
z, and intensity masks propagate those indices without changing the existing
point arrays. Clustering returns membership positions in the intensity-stage
array; those positions are mapped back to raw-LiDAR point indices.

For every cluster, the implementation requires:

```text
len(source_point_indices) == len(cluster_points)
raw_points[source_point_indices] == cluster_points
```

O2 is then defined as:

```text
all associated cluster source_point_indices
-> unique(source_point_index)
-> raw_points[unique_indices]
-> frozen diagnostic PCA
-> GT IoU
```

`np.unique(stacked_coordinate_rows, axis=0)` is no longer used for formal O2.
Distinct raw records with identical coordinates remain distinct PCA samples;
the same source index referenced more than once is retained once.

## Scope

O1, O3-O6, canonical stage point counts, clustering parameters, classifier,
NMS, Evaluation, and TP/FP/FN semantics are unchanged. The focused tests cover
both identity edge cases, union-order determinism, and the unchanged O3-O6
canonical-count gate.

The corrected diagnostic replay will emit a SHA-256 sidecar for the final JSON.
The hash is artifact identification only and is not part of the algorithm or
diagnostic schema.
