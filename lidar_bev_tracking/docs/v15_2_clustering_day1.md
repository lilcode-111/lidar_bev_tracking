# 第 15.2 轮 Day1：聚类协议与配置冻结

本日只冻结协议，不改变现有 A0' detector 的实际聚类结果。

## 已冻结内容

- 距离使用 `range_xy = sqrt(x^2 + y^2)`。
- 距离段为 `[0, 15)`、`[15, 30)`、`[30, +inf)`。
- 自适应配置必须同时声明 near、mid、far 三段的 `eps` 和 `min_points`。
- 全局候选半径上限为 `global_max_eps=0.85`。
- 点对邻域规则冻结为 `pairwise_eps(i,j)=max(eps_i, eps_j)`。
- `min_points` 包含查询点自身。
- Day1 的协议快照使用 `mode=fixed, eps=0.6, min_points=20`，用于后续 C0 精确回放。

## 为什么不在本日直接替换 detector

15.2 首先需要证明新的实现与已冻结的 I0/A0' baseline 一致。若 Day1 就替换旧入口，后续出现差异时无法区分是配置变化、邻域实现变化还是聚类算法变化。因此现有 `euclidean_cluster()` 保持不动，Day2 再加入 brute-force reference、grid 实现和统一调用入口。

## 配置文件

协议快照位于 `configs/experiments/v15/c0_clustering_protocol.yaml`。它是实验协议记录，不是完整 KITTI batch 运行配置。
