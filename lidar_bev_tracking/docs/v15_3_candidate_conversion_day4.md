# v15.3 Day4: Downstream Loss Attribution

本日补充候选下游归因：

- 重建确定性 NMS 的 suppressor/suppressed 关系；
- 记录关联 Car candidate 的中心、长度、宽度、yaw 和 IoU 误差；
- 区分关联候选真正匹配、IoU 足够但未匹配、以及与其他 GT 竞争的情况。

这些函数只读取已有的 raw detection、NMS 结果和 evaluation 结果，不修改任何检测或评测逻辑。
