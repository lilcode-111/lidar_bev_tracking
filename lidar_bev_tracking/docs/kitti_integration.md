# KITTI 点云接入

第五轮新增 KITTI Object Detection LiDAR 点云接入，让当前聚类检测链路可以跑在真实传感器点云上。

默认目录：

```text
data/kitti/
  training/
    velodyne/
      000000.bin
    label_2/
      000000.txt
```

最小可运行只需要：

```text
training/velodyne/*.bin
```

`label_2/*.txt` 是可选的，本轮只做轻量解析，主要为后续评价做准备。

运行：

```bash
PYTHONPATH=src python scripts/run_kitti_clustering_demo.py --frame-id 000000
```

输出：

```text
outputs/figures/kitti_clustering_000000.png
```

当前流程：

```text
KITTI .bin
-> load_kitti_point_cloud()
-> detect_objects_from_points()
-> nms_bev()
-> points_to_bev()
-> draw_objects()
```

本轮暂时不做相机投影和 calib 坐标转换，检测直接在 LiDAR 坐标系中完成。后续可以基于 `label_2` 和 `calib` 增加 GT 可视化、IoU 评价和 KITTI 指标统计。
