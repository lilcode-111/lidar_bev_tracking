# KITTI Calib and GT Overlay

第七轮目标是把 KITTI 的 `label_2` 标注真正用起来。

KITTI 的 LiDAR 点云在 Velodyne 坐标系下：

```text
x: forward
y: left
z: up
```

但 `label_2/*.txt` 里的 3D box 标注在相机坐标系下：

```text
x: right
y: down
z: forward
```

所以不能直接把 label 画到 BEV 上。第七轮新增的流程是：

```text
label_2/*.txt
-> 读取 class / dimensions / location_camera / rotation_y
-> 读取 calib/*.txt
-> 计算 camera rect -> LiDAR 的变换矩阵
-> 转成项目内部 box 格式
-> 和聚类检测框一起画到 BEV 图上
```

运行 mini KITTI sample：

```bash
PYTHONPATH=src python scripts/create_mini_kitti_sample.py --frame-id 000000
```

运行 GT overlay demo：

```bash
PYTHONPATH=src python scripts/run_kitti_gt_overlay_demo.py --frame-id 000000
```

输出图片：

```text
outputs/figures/kitti_gt_overlay_000000.png
```

图中：

```text
det:*  表示聚类检测框
gt:*   表示 KITTI label 转成 LiDAR 坐标系后的 GT 框
```

注意：mini sample 的 calib 是为了 smoke test 写的简化标定，只用于验证代码链路。真实 KITTI 数据需要使用官方 `training/calib/*.txt`。

真实数据目录应为：

```text
data/kitti/
  training/
    velodyne/
      000000.bin
    label_2/
      000000.txt
    calib/
      000000.txt
```

第七轮仍然不是评价指标。它解决的是评价之前的关键问题：检测框和 GT 框必须先处在同一个 LiDAR BEV 坐标系里。
