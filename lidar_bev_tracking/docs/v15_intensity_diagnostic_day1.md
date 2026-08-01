# 第 15.1 轮 Day1：Intensity 诊断实验门禁

## 目标

在运行 I0 / I1 之前冻结实验基线、诊断帧和几何证据，避免把数据变化或坐标问题误判为 intensity 参数收益。

## 本日完成内容

1. 冻结 `A0_prime_geometry_corrected` 基线声明，记录源 run、代码 commit、配置 hash、100 帧指标和检测计数。
2. 固定 100 帧与 25 帧 manifest 的规范化 SHA-256；计算前统一 LF/CRLF 换行，诊断 manifest 必须是 25 个唯一、六位、升序 frame id。
3. 新增 A0' 声明门禁，自动校验基线状态、manifest hash、`intensity_min=0.38`、TP+FN 和 effective detection count 守恒。
4. 将独立 KITTI yaw / corner 语义检查接入 25 帧 FailureEvidence replay。
5. 报告统一落盘 center round-trip、yaw round-trip、yaw semantic、corner alignment 的均值、最大值、容差和最差对象。
6. 多方向角点测试覆盖接近 `-pi`、`-pi/2`、零、`pi/2` 和接近 `pi` 的方向。

## 门禁意义

Day1 不比较 I0 / I1 指标。只有声明、manifest、几何语义和误差汇总全部通过后，后续实验才允许解释 `intensity_min` 的单变量影响。
