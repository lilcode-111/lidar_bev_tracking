# KITTI yaw 语义修复

## 问题

旧实现把 KITTI `rotation_y` 的方向向量写成：

```text
[sin(rotation_y), 0, cos(rotation_y)]
```

这相当于使用相机 z 轴作为车辆长度方向。KITTI 3D box 的 length 轴位于物体局部 x 轴，因此正确方向向量是：

```text
[cos(rotation_y), 0, -sin(rotation_y)]
```

真实帧 `000424` 的 11 个有效框均出现约 `pi/2` 的 yaw 语义误差，独立角点对齐误差最大为 `3.62m`。

## 修复

- camera `rotation_y` 转 LiDAR yaw 时使用 KITTI length 轴；
- LiDAR yaw 转 camera `rotation_y` 时使用对应逆关系；
- 更新旧测试中 `rotation_y=0 -> lidar yaw=0` 的错误预期；
- 保留独立角点语义测试，防止双向公式同时写错却仍通过 round-trip。

## 验证

```bash
PYTHONPATH=src python -m unittest tests.test_kitti_yaw_semantics -v
PYTHONPATH=src python -m unittest tests.test_geometry_sanity -v
PYTHONPATH=src python -m unittest discover -s tests -v
```

真实帧验证：

```bash
PYTHONPATH=src python scripts/run_kitti_yaw_semantic_check.py \
  --data-root data/kitti/real_100 \
  --frame-id 000424
```

修复后要求：

- `KITTI yaw semantics passed: True`；
- 所有有效框通过；
- yaw 语义误差和角点对齐误差落入各自容差；
- 角点默认容差为 `0.001m`，用于覆盖真实标定矩阵和 `float32` 运算产生的亚毫米误差。

修复确认后，需要重新运行固定 25 帧 FailureEvidence 和固定 100 帧 A0。旧报告因 GT 朝向错误而失效，不能继续用于算法 A/B。
