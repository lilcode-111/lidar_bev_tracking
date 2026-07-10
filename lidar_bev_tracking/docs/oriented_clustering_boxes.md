# Oriented Clustering Boxes

第九轮目标是优化聚类检测框。

之前的聚类检测框是轴对齐框：

```text
x_min / x_max / y_min / y_max
-> length / width
-> yaw = 0
```

这种方法能从点云簇生成检测框，但框只能横平竖直，不能估计目标朝向。

第九轮新增 PCA oriented box：

```text
cluster 点云簇
-> 取 xy 平面点
-> PCA 找点云主方向
-> 用主方向作为 yaw
-> 在局部坐标系下计算 length / width
-> 输出有朝向的 detection box
```

运行 demo：

```bash
PYTHONPATH=src python scripts/run_oriented_clustering_demo.py
```

输出图片：

```text
outputs/figures/oriented_clustering_detection.png
```

也可以在 KITTI evaluation demo 中使用有向框：

```bash
PYTHONPATH=src python scripts/run_kitti_eval_demo.py --frame-id 000000 --oriented
```

对比普通轴对齐框：

```bash
PYTHONPATH=src python scripts/run_kitti_eval_demo.py --frame-id 000000
```

注意：PCA oriented box 仍然是传统点云几何方法，不是深度学习检测器。它适合作为 baseline，帮助理解点云聚类、目标主方向估计和 BEV IoU 评价之间的关系。
