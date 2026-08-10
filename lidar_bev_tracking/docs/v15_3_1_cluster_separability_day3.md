# v15.3.1 Cluster Separability Diagnostic — Day 3

Day 3 consumes the already-built Day 1 grouping and compact Day 2 feature result.
It does not rerun grouping, delta marking, feature extraction, or any detector stage.

The final `cluster_separability_diagnostic` contains:

- C1 P1 and P2 distributions by near/mid/far/total;
- unique strict-background N clusters from the same frame and distance bin;
- a focused delta-22 P2 versus matched-N comparison;
- P2 and delta-22 oracle-IoU summaries;
- compact per-cluster delta-22 evidence.

Statistics are count, min, P25, median, mean, P75, and max. The final JSON stores
only the compact Day 3 diagnostic, avoiding repeated Day 1/Day 2 record payloads.
No clustering, classifier threshold, NMS, or Evaluation setting is changed.
