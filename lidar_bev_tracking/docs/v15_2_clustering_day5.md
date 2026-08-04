# 第 15.2 轮 Day5：variant registration 与 merging gate

本日新增实验变体登记和 GT-cluster 关联门禁。

## C0/C1/C2/C3

四个 variant 必须按固定顺序登记。C0 强制为：

```text
mode=fixed
eps=0.60
min_points=20
intensity_min=0.38
z_min=-0.9
```

所有 variant 的 `intensity_min`、`z_min` 必须保持不变。C1/C2/C3 的具体距离段参数必须在评审确认后填入，不能根据结果临时修改。

## GT-cluster 关联

只处理 positive Car GT。对每个 GT：

```text
N_g = GT 框内的 intensity-filtered 点数
eligible ⇔ N_g >= 3
n_cg >= max(3, ceil(0.10 * N_g))
```

一个 cluster 关联至少两个 eligible positive GT 时，标记为 merged cluster。`merged_positive_gt_count` 统计受影响的唯一 `(frame_id, gt_id)`，不是 cluster 数量。

## 预注册边界

`preregister_variant_specs()` 只记录配置和冻结字段，不根据指标自动挑选参数。所有 variant 的 eligible GT 集合必须与 C0 一致，否则实验直接不通过。

Day6 将使用这些 variant report 汇总 TP/FP/FN、F1、zero-detection 和 merging rate，并按固定规则输出诊断排序。排序结果只能用于分析，不能替代四方评审对 release candidate 的确认。
