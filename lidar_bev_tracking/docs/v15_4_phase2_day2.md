# 15.4 Phase-2 Day2: authorized T0 replay

Day2 adds the real T0-only execution path. It runs the frozen 100-frame T0
configuration, derives the frozen delta-22 25-frame reference view, and checks
both views against the materialized references at absolute tolerance `1e-8`.
No non-T0 variant is executed. One permanent entry point supports the mutually
exclusive `--authorize`, `--plan-only`, and `--execute-t0` modes. Commit Day2,
regenerate authorization through that runner, then execute T0 through it.
