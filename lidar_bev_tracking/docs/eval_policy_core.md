# 第 12.1 轮：Evaluation Policy Consistency Fix

本轮目标是让代码实现、测试计划、报告结构和评测声明保持一致。

不做大重构，不做异常隔离，不做 run_id，不做 failure visualization。

## 评测类别口径

当前仍然是 Car-only BEV evaluation：

```text
positive GT:
  Car

neutral GT:
  Van
  Truck

excluded GT:
  Pedestrian
  Cyclist
  Person_sitting
  Tram
  Misc

dontcare:
  DontCare
```

detection 侧：

```text
positive detection:
  car detection 且中心点在 ROI 内

ignored detection:
  非 car detection
  或 ROI 外 detection
```

ROI：

```text
x in [0, 40)
y in [-20, 20)
```

## IoU 阈值

主指标：

```text
positive matching threshold = 0.5
neutral suppression threshold = 0.5
```

辅助指标：

```text
positive matching threshold = 0.25
neutral suppression threshold = 0.5
```

也就是说，positive 阈值跟随当前评测阈值，neutral 阈值固定为 0.5。

这样做是为了避免辅助 IoU=0.25 时，Van / Truck 过于宽松地吸收 detection，导致 FP 被过度 neutralize。

## TP / FP / FN / Neutralized

匹配顺序：

```text
1. detection 按 score 降序、det_index 升序排序；
2. car detection 优先匹配 unmatched positive Car GT；
3. 如果没有匹配到 Car GT，再尝试匹配 unmatched neutral Van / Truck GT；
4. 命中 positive GT：计 TP；
5. 命中 neutral GT：计 neutralized，不计 TP，也不计 FP；
6. 同一个 neutral GT 只能吸收一个 detection；
7. 同一 neutral GT 上第二个重复 detection 计 FP；
8. car detection 没命中 positive 或 neutral：计 FP；
9. unmatched positive Car GT：计 FN；
10. excluded / dontcare / ROI 外 GT 不计 FN。
```

## 指标

单帧 metrics、per-class metrics、batch summary、CSV 都输出：

```text
tp
fp
fn
precision
recall
f1
```

F1 使用 counts 直接计算：

```text
F1 = 2TP / (2TP + FP + FN)
```

这样可以避免 precision 或 recall 为 null 时再反推 F1 的歧义。

## 零分母

如果 denominator 为 0：

```text
JSON: null
CSV: 空字段
终端: undefined
```

例如：

```text
无 detection 且无 GT:
  precision = null
  recall = null
  f1 = null
```

## Batch 双阈值输出

batch summary 会显式输出：

```json
"metrics_by_iou": {
  "0.50": {
    "tp": 10,
    "fp": 0,
    "fn": 0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  },
  "0.25": {
    "tp": 10,
    "fp": 0,
    "fn": 0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  }
}
```

逐帧 CSV 会展开为：

```text
tp_iou_0_50
fp_iou_0_50
fn_iou_0_50
precision_iou_0_50
recall_iou_0_50
f1_iou_0_50
tp_iou_0_25
fp_iou_0_25
fn_iou_0_25
precision_iou_0_25
recall_iou_0_25
f1_iou_0_25
```

## 验收测试清单

第 12.1 轮需要补充的阻塞测试：

```text
1. positive 优先；
2. neutral 固定阈值；
3. excluded / DontCare；
4. ROI 边界；
5. zero-denominator 矩阵；
6. batch 双阈值汇总测试。
```
