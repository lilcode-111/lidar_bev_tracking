# KITTI BEV Evaluation

第八轮目标是把检测结果和 GT 标注框放到同一个 LiDAR BEV 坐标系下做基础评价。

前置流程：

```text
KITTI label_2 + calib
-> 转成 LiDAR 坐标系 GT box

KITTI velodyne 点云
-> 聚类检测
-> NMS
-> detection box
```

评价流程：

```text
detection box + GT box
-> 按类别匹配
-> 计算 BEV IoU
-> 判断 TP / FP / FN
-> 统计 precision / recall
```

运行 mini KITTI sample：

```bash
PYTHONPATH=src python scripts/create_mini_kitti_sample.py --frame-id 000000
```

运行 BEV 评价 demo：

```bash
PYTHONPATH=src python scripts/run_kitti_eval_demo.py --frame-id 000000
```

输出报告：

```text
outputs/reports/kitti_eval_000000.json
```

核心指标含义：

```text
TP：检测框和同类别 GT 框匹配成功，并且 IoU >= 阈值
FP：检测框没有匹配到合格 GT，属于误检
FN：GT 没有被任何检测框匹配到，属于漏检
```

计算方式：

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
```

当前默认评价阈值：

```text
eval_iou_threshold = 0.25
```

这个阈值偏低，是因为当前检测器还是简单聚类 baseline，输出框比较粗糙，mini sample 也是为了验证流程的简化数据。真实 KITTI benchmark 后续需要使用更严格的评价协议和更合理的 IoU 阈值。

第八轮的重点不是追求高指标，而是把完整评价闭环跑通：

```text
真实/mini 点云
-> 检测框
-> GT 框坐标转换
-> IoU 匹配
-> TP / FP / FN
-> precision / recall
```
