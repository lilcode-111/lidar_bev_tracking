# 第 15.2 轮 Day3：统一聚类入口与 C0 回放基础

本日新增 `cluster_points()` 统一入口：

- `mode=fixed`：调用原有 `euclidean_cluster()`，保留 A0'/I0 的历史行为；
- `mode=adaptive`：调用 Day2 的 brute-force 或 adaptive grid 实现。

`detect_objects_from_points()` 新增可选的 `clustering_policy` 参数。参数不传时，默认路径完全不变；传入 fixed policy 时，检测输出必须与默认路径逐字段一致。

## C0 的含义

C0 使用固定参数 `eps=0.6, min_points=20, intensity_min=0.38`。Day3 的单测验证：

1. 统一 fixed 入口与 legacy cluster 的数量、顺序和点成员完全一致；
2. detector 默认路径和 fixed policy 路径输出完全一致；
3. adaptive brute-force 与 adaptive grid 的 cluster 结果一致。

这为后续 KITTI 25 帧或 100 帧 C0 replay 提供了接口基础，但本日尚未执行真实数据回放，也没有改变 batch pipeline 的默认调用。
