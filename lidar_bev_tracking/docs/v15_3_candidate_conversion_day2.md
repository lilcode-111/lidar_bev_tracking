# v15.3 Day2: Candidate Lineage Assembly

本日将 Day1 协议接到已完成的 replay 中间结果，组装每个 positive GT 的完整血缘：

`GT -> associated clusters -> raw detections -> Car candidates -> NMS -> IoU/evaluation`

关联规则沿用 v15.2：GT 内有效点数至少为 3，且 cluster 在 GT 内的点数达到 `max(3, ceil(0.10 * N_g))`。所有满足条件的 cluster 都保留，不只选择 best cluster。

评测匹配只接受与当前 GT 具备血缘关系的 Car detection。即使全帧存在一个无血缘检测框与 GT 重合，也不能把该 GT 归因为 `matched_at_0_50`。

本日仍不修改聚类、分类、NMS、PCA 或评测参数；下一步才会把该组装逻辑接入 C0/C1 批量 replay。
