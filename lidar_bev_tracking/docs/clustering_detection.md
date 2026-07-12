# 点云聚类检测 Demo

第四轮新增的是一个 LiDAR 感知检测 baseline：不再用 GT 人工生成预测框，而是直接从点云中检测障碍物。

整体流程：

```text
LiDAR points
-> ROI 过滤
-> 障碍物点过滤
-> 欧式聚类
-> cluster 拟合 BEV box
-> NMS
-> 可视化检测框
```

核心区别：

```text
前三轮: GT objects -> noisy predictions -> NMS
第四轮: point cloud -> clustering detector -> detections -> NMS
```

当前检测器使用简单规则：

```text
1. 保留车辆前方 0~40m、左右 -20~20m 的点
2. 过滤低高度、低反射强度背景点
3. 按 x/y 距离做欧式聚类
4. 对每个 cluster 取 min/max 拟合 axis-aligned BEV box
5. 根据尺寸和点数粗略判断 car / pedestrian / cone
```

它不是深度学习检测器，但已经体现了 LiDAR 感知中的核心链路：

```text
点云预处理
障碍物点提取
点云聚类
目标框拟合
检测后处理
```

这里的 synthetic.py 仍然属于合成数据/简化仿真部分；clustering_detector.py 才是本轮新增的感知检测模块。

后续可以升级：

```text
1. 用 PCA 拟合带 yaw 的旋转框
2. 接入真实 KITTI / nuScenes 点云
3. 将聚类检测替换为 PointPillars / CenterPoint 等深度学习检测器
```
