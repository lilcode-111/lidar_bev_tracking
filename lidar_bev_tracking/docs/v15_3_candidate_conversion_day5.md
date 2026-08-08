# v15.3 Day5: Replay Integration and Final Attribution Output

本日把 Day2-Day4 归因能力接入现有的 `run_variant_frame` replay。

当 replay 使用 `candidate_variant="C0"` 或 `candidate_variant="C1"` 时，单帧报告会新增 `candidate_conversion`，包含：

- 每个 positive GT 的完整 cluster/detection 血缘；
- 唯一 terminal state；
- NMS suppressor 关系；
- box geometry 误差；
- Evaluation 匹配和竞争信息。

批量诊断的 `diagnostics.<variant>.candidate_conversion_analysis` 汇总 waterfall、分类特征、距离段和终态计数。

不传 `candidate_variant` 时，旧版 replay 输出保持不变；本日没有修改任何算法阈值。
