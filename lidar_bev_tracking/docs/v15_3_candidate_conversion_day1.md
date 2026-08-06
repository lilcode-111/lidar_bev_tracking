# v15.3 Day1: Candidate Conversion Protocol

本日只冻结 15.3 的归因协议和数据结构，不接入检测主流程，也不修改 C0/C1 参数。

## 固定链路

`filtered points -> associated cluster -> raw detection -> Car classification -> NMS -> IoU -> evaluation`

每个 positive GT 必须且只能有一个 `CandidateConversionState`。

## 终态优先级

1. `no_filtered_points`
2. `insufficient_filtered_points_for_association`
3. `no_associated_cluster`
4. `rejected_by_car_classifier`
5. `removed_by_nms`
6. `box_iou_below_0_25`
7. `box_iou_0_25_to_0_50`
8. `iou_ge_0_50_but_unmatched`
9. `matched_at_0_50`

后续 Day2 才会把 C0/C1 replay 结果填入这些结构，并保存完整 cluster/detection 分支血缘。
