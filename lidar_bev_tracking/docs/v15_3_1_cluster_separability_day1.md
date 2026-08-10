# v15.3.1 Cluster Separability Diagnostic — Day 1

Day 1 adds read-only cluster grouping only. It does not change clustering,
classification, NMS, evaluation, or any frozen threshold.

## Groups

- `P1`: positive-Car-GT-associated cluster accepted as a Car candidate before NMS.
- `P2`: positive-Car-GT-associated cluster rejected by the Car classifier.
- `N`: cluster without a positive-Car-GT association. Any overlap with neutral,
  excluded, DontCare, or below-gate positive annotations remains explicit so Day 2
  can select a strict background comparison set.

The unit of analysis is a unique cluster. All associated GT ids and overlap counts
are retained. C1 clusters associated with the C0-to-C1 delta GT cohort receive
`is_delta_22=true`; the marker is restricted to P2.

## Day 1 gates

- Every cluster appears exactly once in P1, P2, or N.
- P1/P2 must have a positive Car GT association; N must not.
- Cluster ids are unique within a frame.
- The C0/C1 candidate-conversion GT sets must match before delta marking.
