# 第 12 轮：Evaluation Policy Core

第 12 轮的目标不是提高 Precision / Recall，而是冻结一版更可信的评测口径。

核心原则：

```text
parser 只负责忠实读取数据
policy 决定哪些类别参与评测
evaluation 按固定规则计算 TP / FP / FN
```

## KITTI Label 读取口径

`kitti.py` 不再提前丢弃评测逻辑暂时不用的类别。

现在 parser 会保留：

```text
Car
Van
Truck
Pedestrian
Person_sitting
Cyclist
Tram
Misc
DontCare
```

其中：

```text
缺 label 文件：抛 FileNotFoundError
空 label 文件：合法零 GT，返回 []
```

这样可以避免把“数据缺失”误当成“这一帧确实没有 GT”。

## 当前 Car-only Policy

当前评测冻结为 Car-only：

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
  DontCare 保留并统计，但没有有效 3D box 时不参与 BEV 匹配
```

detection 侧：

```text
只有 car detection 进入 Car evaluation
pedestrian / cone / cyclist detection 不进入 Car evaluation
```

ROI：

```text
x in [0, 40)
y in [-20, 20)
```

GT 和 detection 都按中心点是否在 ROI 内判断。

## TP / FP / FN 定义

主指标：

```text
IoU = 0.5
```

辅助指标：

```text
IoU = 0.25
```

两个阈值会独立匹配，不复用彼此结果。

匹配顺序：

```text
1. 按 score 降序、det_index 升序排序 detection
2. 每个 car detection 先匹配 unmatched positive Car GT
3. 如果没有匹配到 positive GT，再尝试匹配 unmatched neutral Van / Truck GT
4. 匹配到 positive GT：计 TP
5. 匹配到 neutral GT：neutralized，不计 TP，也不计 FP
6. 一个 neutral GT 只能吸收一个 detection
7. 同一 neutral GT 上第二个重复 detection 计 FP
8. 没匹配到 positive 或 neutral 的 car detection 计 FP
9. unmatched positive Car GT 计 FN
```

## 稳定排序

为了保证同样输入每次结果一致，detection 会带：

```text
det_index
```

排序规则：

```text
score 降序
det_index 升序
```

NMS 和 evaluation 都使用这套规则。

## 零分母

如果 precision 或 recall 的分母为 0，不再写成 `0.0`，而是写成：

```json
null
```

终端打印时显示：

```text
undefined
```

这样可以区分：

```text
指标无法定义
```

和：

```text
指标真的为 0
```

## 最小测试覆盖

本轮新增测试覆盖：

```text
1. KITTI 类别保留
2. 缺 label 文件和空 label 文件区分
3. Van / Truck neutral 逻辑
4. 非 car detection 不进入 Car evaluation
5. IoU=0.5 和 IoU=0.25 独立匹配
6. 零分母指标为 null
7. det_index 保证 evaluation 稳定
8. det_index 保证 NMS 稳定
```

运行：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
