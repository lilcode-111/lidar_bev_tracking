# 第 15.1 轮 Day2：I0/I1 单变量实验门禁

## 目标

在固定25帧诊断集上运行 I0 与 I1，只允许 `detector.intensity_min` 从 `0.38` 变化为 `0.0`。I0 必须逐帧精确复现 A0' 对应帧后，I1 才允许运行和解释。

## 配置

- I0：`configs/experiments/v15/i0_intensity_038.yaml`
- I1：`configs/experiments/v15/i1_intensity_000.yaml`

两份 effective config 的唯一差异必须是：

```text
detector.intensity_min: 0.38 -> 0.0
```

## 自动门禁

1. A0' declaration 与100帧源 run 完整、可读取；
2. 25帧 manifest 的 hash、数量和顺序一致；
3. I0/I1 配置只有 `intensity_min` 不同；
4. I0 的双 IoU 指标与关键检测计数逐帧复现 A0'；
5. 每帧满足 `raw >= roi >= z_filter >= intensity_filter`；
6. 每个 IoU 满足 `TP + FN = positive GT`；
7. effective Car detection count 满足 `TP + FP + neutralized`；
8. I1 每帧的 z-filter 与 intensity-filter 点数组完全相同。

## 输出

```text
outputs/intensity_diagnostic/i0_intensity_038.json
outputs/intensity_diagnostic/i1_intensity_000.json
outputs/intensity_diagnostic/i0_i1_comparison.json
```

25帧结果仅用于判断 intensity filter 假设，不作为产品正式指标。
