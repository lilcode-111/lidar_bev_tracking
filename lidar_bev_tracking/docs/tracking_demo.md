# 多目标跟踪 Demo 逻辑

第三轮新增的是一个简化版 tracking-by-detection 流程。

整体流程：

```text
生成多帧目标运动
-> 每帧生成模拟检测框
-> NMS 删除重复框
-> 用 BEV IoU 关联 detection 和已有 track
-> 维护 track id 和轨迹历史
-> 输出逐帧可视化图片
```

当前版本先使用 BEV IoU 做贪心匹配：

```text
1. 对每个已有 track 做简单位置预测
2. 计算 predicted track box 和 detection box 的 BEV IoU
3. 同类别且 IoU 超过阈值时认为可以匹配
4. 优先选择 IoU 最大的匹配
5. 没匹配上的 detection 新建 track
6. 没匹配上的 track 允许短时间 missed
```

这不是最终工业级 tracker，但已经覆盖了多目标跟踪的核心概念：

```text
detection
association
track id
missed count
track lifecycle
trajectory history
```

后续可以继续升级为 Kalman Filter 预测和 Hungarian Matching 全局最优匹配。
