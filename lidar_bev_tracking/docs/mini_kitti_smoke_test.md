# Mini KITTI Smoke Test

这个脚本用于在不下载完整 KITTI 数据集的情况下，快速验证 KITTI 点云读取链路是否能跑通。

它会生成一个很小的 KITTI 风格目录：

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

其中：

```text
velodyne/000000.bin    保存点云，格式是 [x, y, z, intensity]
label_2/000000.txt     保存 KITTI 风格标签
calib/000000.txt       保存简化标定文件，用于 GT overlay smoke test
```

生成 mini sample：

```bash
PYTHONPATH=src python scripts/create_mini_kitti_sample.py --frame-id 000000
```

运行 KITTI 聚类检测 demo：

```bash
PYTHONPATH=src python scripts/run_kitti_clustering_demo.py --frame-id 000000
```

运行 GT overlay demo：

```bash
PYTHONPATH=src python scripts/run_kitti_gt_overlay_demo.py --frame-id 000000
```

期望看到类似输出：

```text
loaded points: ...
loaded labels: ...
gt boxes in lidar frame: ...
raw detections: ...
detections after nms: ...
saved outputs/figures/kitti_gt_overlay_000000.png
```

注意：这个 mini sample 是合成点云写成的 KITTI 文件格式，只用于验证文件路径、`.bin` 解析、label 解析、calib 解析、坐标转换、聚类检测、NMS 和 BEV 可视化流程。

它不是真实 KITTI 数据，也不能作为最终检测指标或 benchmark 结果。

后续如果使用真实 KITTI 数据，需要把真实文件放到：

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
