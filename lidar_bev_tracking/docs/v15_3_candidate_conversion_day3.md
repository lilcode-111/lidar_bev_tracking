# v15.3 Day3: Waterfall and Classifier Feature Audit

本日新增只读分析函数，按 `near_0_15`、`mid_15_30`、`far_30_inf` 和 `total` 输出：

- GT 过滤、cluster、Car candidate、NMS、IoU 和 Evaluation waterfall；
- 关联 cluster 的点数、轴对齐尺寸、height、距离、点密度分布；
- `max(axis_length, axis_width) >= 2.0` 与 `num_points >= 500` 两个现有分类条件的通过统计；
- 各距离段的 terminal state 数量。

本日只统计现有规则，不修改 `2.0m` 或 `500 points` 门槛，不执行参数搜索。若结果证明分类规则是主要瓶颈，下一步应单独预注册 15.3.1 分类消融。
