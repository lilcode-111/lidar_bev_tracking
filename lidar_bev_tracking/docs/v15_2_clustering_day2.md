# 第 15.2 轮 Day2：brute-force 参考与 adaptive grid

本日实现两个邻域查询版本：

- `brute_force_neighbors()`：逐点检查所有点，是 correctness oracle；
- `adaptive_grid_neighbors()`：使用 `global_max_eps=0.85` 的网格缩小候选范围。

两者都执行同一条冻结规则：

```text
distance(i, j) <= max(eps_i, eps_j) + 1e-6
```

邻域结果必须满足：

- 查询点自身出现且只出现一次；
- 邻域关系对称；
- 邻居索引按升序输出；
- adaptive grid 与 brute-force 逐点结果完全一致。

本日没有把新实现接入旧 detector。这样做是为了让后续 C0 能逐项对账：先证明新邻域实现正确，再证明固定参数下它能复现 I0/A0'。
