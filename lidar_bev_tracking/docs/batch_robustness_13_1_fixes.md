# 第 13.1 轮：Batch Robustness Core 合入前一致性修正

第 13.1 轮不是新增功能，而是对第 13 轮 Batch Robustness Core 的合入前一致性修正。

## 修正目标

本轮修正以下 P0 问题：

1. `FrameMetrics` 恢复 `per_class`，避免第 12.1 轮已验收的按类别指标输出退化；
2. pipeline 内部对 missing input 的状态与 batch precheck 保持一致；
3. `BatchResult` 补齐 `started_at`、`finished_at`、`total_time_ms`；
4. `report_writer` 调整写出顺序，保证 `manifest`、`frames.csv`、`summary.json` 对同一帧状态一致；
5. `frame_report_path` 只在 per-frame JSON 写成功后记录，避免引用不存在文件；
6. 必要报告产物写失败时使用稳定 `ReportWriteError` 和 `ErrorCode`。

## 状态语义

missing input 统一为：

```text
缺 bin   -> skipped + missing_bin
缺 label -> skipped + missing_label
缺 calib -> skipped + missing_calib
文件存在但无法读取或解析失败为：
bin 读取失败    -> failed + bin_read_failed
label 解析失败  -> failed + label_parse_failed
calib 解析失败  -> failed + calib_parse_failed
per_class 保留
FrameMetrics 新增：
per_class: dict
单帧 evaluation 中的 metrics["per_class"] 会保留到结构化 FrameResult。
batch 聚合时先累计各类别的 tp/fp/fn，再重新计算：
precision
recall
f1
避免简单平均导致指标不准确。
report_writer 写出顺序
最终写出顺序为：
1. 创建 run 目录
2. 写 config_input.yaml / config_effective.json
3. 写 git.json
4. 写 frames/*.json
5. 根据 per-frame JSON 写出结果更新 FrameResult 状态
6. 重建最终 BatchResult
7. 写 frame_manifest.json
8. 写 frames.csv
9. 最后写 summary.json
这样可以保证：
frame_manifest.json
frames.csv
summary.json
对同一帧状态一致。
frame_report_path 规则
frame_report_path 只在 per-frame JSON 成功写出后写入。
如果某帧 JSON 写失败：
该帧状态降级为 partial_success
记录 frame_report_write_failed
不保留不存在的 frame_report_path
必要报告失败
必要产物包括：
config_effective.json
git.json
frame_manifest.json
frames.csv
summary.json
这些文件写失败时，会抛出：
ReportWriteError
并包含：
error_code
output_path
cause