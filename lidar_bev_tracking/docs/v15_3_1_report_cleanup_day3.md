# v15.3.1 Report Cleanup Day 3

## Contract cleanup

- `ranking` contains only rank, variant, selection values, and sort rule.
- The eligible-GT gate stores one C0 reference list plus counts, hashes, and
  mismatches for the other variants.
- Distance metrics are included in the canonical conversion waterfall instead
  of being emitted as a second distance-analysis tree.
- Diagnostics declare `schema_version: 15.3.1` and
  `candidate_conversion.evidence` as their source of truth.

## Compatibility

Legacy `gt_candidate_records` can still be read by the low-level distance
helper for old replay tests. New ablation reports do not emit that duplicate
structure. `primary_reason_counts` remains available because its filter-reason
semantics are different from candidate-conversion terminal states.
