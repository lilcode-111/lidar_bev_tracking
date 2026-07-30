# 第 15 轮 Day 2：坐标与 3D GT Sanity Check

## 目标

在进行 intensity 和 clustering A/B 前，先验证 KITTI camera、rectified camera 与 LiDAR 坐标转换没有系统性错误。

Day 2 不修改：

```text
Evaluation Policy
BEV IoU
ROI
聚类参数
PCA box
NMS
分类规则
```

## GT 3D box

KITTI label 中的 `location_camera` 是目标底面中心。转换到 LiDAR 后，本项目使用目标几何中心，因此先执行：

```text
center_camera_y = location_camera_y - height / 2
```

LiDAR GT box 现在完整保留：

```text
x, y, z, length, width, height, yaw
```

`height` 只供 3D point-in-box 和 FailureEvidence 使用，不参与现有 BEV IoU，因此不会改变 A0 的 TP/FP/FN 口径。

## Sanity 检查

单帧检查包括：

1. LiDAR 点变换到 rectified camera 后再变回 LiDAR；
2. camera `rotation_y` 转成 LiDAR `yaw` 后再转回；
3. GT 中心转到 LiDAR 后再恢复 KITTI 底面中心；
4. `length/width/height` 是否保持；
5. 每个 Car GT 的 oriented 3D box 内是否存在 LiDAR 点。

运行示例：

```bash
PYTHONPATH=src python scripts/run_geometry_sanity.py \
  --data-root data/kitti/real_100 \
  --frame-id 000424
```

输出：

```text
outputs/geometry_sanity/geometry_sanity_000424.json
```

坐标 round-trip 失败时，应停止算法 A/B，并建立独立 bugfix 分支重新生成 baseline。
