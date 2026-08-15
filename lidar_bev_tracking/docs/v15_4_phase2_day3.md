# 15.4 Phase-2 Day3: formal matrix execution

Day3 extends the single formal runner with `--execute-matrix`. The mode requires
an authorized clean commit and a PASS T0 replay gate. It then runs T1, T2, and
T_off in the frozen order, writes each raw report immediately, validates source
point monotonicity, and produces TP/candidate/regression-reason identity audits.
T_off remains diagnostic-only. No threshold or gate can be overridden by CLI.

If all raw reports are valid but the in-memory audit is killed, `--resume-audit`
performs read-only recovery. It loads one report at a time, writes compact NumPy
point-index caches, releases the report, and checks monotonicity per frame. The
raw formal comparison commit and the later audit-recovery commit are both
recorded; no algorithm variant is rerun.
