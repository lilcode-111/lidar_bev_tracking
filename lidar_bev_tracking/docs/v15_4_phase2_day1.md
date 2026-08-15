# 15.4 Phase-2 Day1: authorization and formal plan

Day1 adds post-commit authorization and a plan-only formal runner. It does not
run T0, T1, T2, or T_off. Authorization binds a clean full Git commit, frozen
schedule and release-gate files, data identity, and materialized T0 references.

Commit these files first. On the clean committed HEAD, generate the ignored
authorization artifact, then validate the plan with `--plan-only`. Non-T0
interpretation remains blocked until both 25-frame and 100-frame T0 replay
checks pass.
