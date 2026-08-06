# 第 15.2 轮 Day8：逐帧与距离段诊断输出

本日只增强结果解释，不改变 C0/C1/C2/C3 参数和算法。

`c0_c3_comparison.json` 新增：

- `diagnostics.<variant>.frames`：逐帧双 IoU 指标、candidate generation 和 FN 原因；
- `diagnostics.<variant>.gt_records`：逐 GT 距离段、点数、cluster、Car candidate、best IoU 和 outcome；
- `diagnostics.<variant>.distance_analysis`：near/mid/far 的 TP/FN、coverage、zero-detection 和 outcome 统计。
