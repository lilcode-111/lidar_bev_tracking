# KITTI yaw 语义验证

本验证不修改现有 yaw 转换，只检查当前实现是否符合 KITTI 3D 标注的几何语义。

验证包含三层：

1. 对照测试：现有 camera -> LiDAR -> camera yaw round-trip 仍然可以通过；
2. 语义测试：KITTI `rotation_y=0` 的车辆长度轴应沿相机 x 轴，转换到标准 LiDAR 坐标后应得到 `yaw=-pi/2`；
3. 角点测试：独立构造 KITTI 3D box 四个 BEV 角点，转换到 LiDAR 后，与项目生成的 GT box 四角比较。

先运行定向测试：

```bash
PYTHONPATH=src python -m unittest tests.test_kitti_yaw_semantics -v
```

当前实现若存在 90 度语义偏差，预期结果是：round-trip 对照测试通过，两个语义测试失败。这是验证阶段的预期现象。

再运行真实帧检查：

```bash
PYTHONPATH=src python scripts/run_kitti_yaw_semantic_check.py \
  --data-root data/kitti/real_100 \
  --frame-id 000424
```

脚本输出：

- `yaw_semantic_error_rad`：当前 yaw 与独立长度轴转换结果的角度误差；
- `corner_alignment_error_m`：当前 GT box 四角与独立 KITTI 角点转换结果的最大匹配误差；
- `passed`：两项误差是否均低于阈值。

真实帧检查失败时脚本返回非零退出码，并写出本地诊断文件：

```text
outputs/geometry_sanity/kitti_yaw_semantics_000424.json
```

该 JSON 是运行产物，不提交 Git。确认问题后，再在独立 bugfix 提交中修改 yaw 转换，并要求同一组测试全部通过。
