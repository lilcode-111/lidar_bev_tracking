# 点云到 BEV 的构建逻辑

当前项目中，一帧点云由两部分组成：

```text
points = background + object_points
```

每个点的格式是：

```text
[x, y, z, intensity]
```

其中 `x` 表示车辆前方距离，`y` 表示车辆左右位置，`z` 表示高度，`intensity` 表示雷达反射强度。

背景点 `background` 用来模拟地面和环境噪声：

```python
background = np.column_stack([
    rng.uniform(0, 40, 6000),
    rng.uniform(-20, 20, 6000),
    rng.uniform(-1.6, 0.3, 6000),
    rng.uniform(0.05, 0.35, 6000),
])
```

含义是生成 6000 个背景点：`x` 在 0~40m，`y` 在 -20~20m，`z` 在 -1.6~0.3m，`intensity` 在 0.05~0.35。

障碍物点先由目标框定义，例如一辆车的中心位置、长宽和朝向：

```text
x=12m, y=3m, length=4.5m, width=1.9m, yaw=0.15
```

然后在目标框内部采样局部点，再根据 `yaw` 旋转并平移到世界坐标：

```python
x = obj_x + cos(yaw) * local_x - sin(yaw) * local_y
y = obj_y + sin(yaw) * local_x + cos(yaw) * local_y
```

点云转 BEV 的核心是：

```text
1. 过滤车辆前方 0~40m、左右 -20~20m 的点
2. 按 0.1m 分辨率建立 400x400 网格
3. 将 x/y 坐标映射到 row/col
4. 每个网格保留最大 intensity
5. 转成 0~255 的灰度图片
```

一句话概括：BEV 就是把 3D 点云投影到 x-y 平面，并离散成固定分辨率的二维栅格图。

当前版本是简化实现，后续可以优化背景点分布、去除障碍物区域内的背景点、模拟表面采样和 LiDAR 射线遮挡。
