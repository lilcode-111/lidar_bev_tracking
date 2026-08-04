# 第 15.2 轮 Day6：消融结果汇总框架

本日补充 C0/C1/C2/C3 的批量结果汇总：

- 主指标使用 IoU=0.50；
- 辅助指标保留 IoU=0.25；
- 聚合 TP/FP/FN、Precision/Recall/F1；
- 聚合 zero-detection-with-GT 和 merging rate；
- 检查四个 variant 的 eligible positive GT 集合完全一致；
- 使用固定排序：TP 降序、FP 升序、merging rate 升序、variant 名称升序。

本日仍不写入 C1/C2/C3 的具体数值。参数必须先完成预注册，再运行固定 25 帧诊断集，最后才进入 100 帧正式验收。
