# v15.3 Review Supplement Day 2

The review supplement now emits a complete terminal-state matrix for every
variant and every frozen distance bin, including `total`. It also emits a
minimal downstream summary for associated Car candidates after NMS:

- geometry candidate count and mean center/size/yaw errors;
- evaluation competition count;
- IoU >= 0.50 but unmatched count;
- matched-at-0.50 count.

This is read-only analysis. It does not change clustering, classification,
NMS, box generation, or evaluation policy.
