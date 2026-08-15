# 15.4 Phase-2 Day3: formal matrix execution

Day3 extends the single formal runner with `--execute-matrix`. The mode requires
an authorized clean commit and a PASS T0 replay gate. It then runs T1, T2, and
T_off in the frozen order, writes each raw report immediately, validates source
point monotonicity, and produces TP/candidate/regression-reason identity audits.
T_off remains diagnostic-only. No threshold or gate can be overridden by CLI.
