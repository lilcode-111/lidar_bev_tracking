# v15.3.1 Day1: Report Schema Freeze

本日冻结报告协议，不删除历史字段。

## 唯一事实来源

单帧 GT 级完整证据统一以：

`candidate_conversion.evidence`

为唯一事实来源。waterfall、分类审计和终态统计都必须由该列表派生。

## 过渡字段

以下字段暂时保留用于兼容旧报告，但不再作为 15.3.1 新逻辑的事实来源：

- `gt_candidate_records`
- `failure_evidence`

## Day1 校验

- schema version 固定为 `15.3.1`；
- 每个 `(frame_id, gt_id)` 只能出现一次；
- `num_positive_gt` 必须等于 evidence 数量；
- 不改变 detection、NMS、evaluation 或指标结果。
