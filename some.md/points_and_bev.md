# 点云到 BEV 的构建逻辑

当前项目中，一帧点云由两部分组成：

```text
points = background + object_points
每个点的格式是：
[x, y, z, intensity]
其中：
x: 车辆前方距离
y: 车辆左右位置
z: 高度
intensity: 雷达反射强度
1. 点云怎么生成
背景点 background 用来模拟地面和环境噪声：
background = np.column_stack([
    rng.uniform(0, 40, 6000),
    rng.uniform(-20, 20, 6000),
    rng.uniform(-1.6, 0.3, 6000),
    rng.uniform(0.05, 0.35, 6000),
])
含义是生成 6000 个背景点：
x: 0~40m
y: -20~20m
z: -1.6~0.3m
intensity: 0.05~0.35
障碍物点 object_points 先由目标框定义，例如一辆车：
中心位置: x=12m, y=3m
尺寸: length=4.5m, width=1.9m
朝向: yaw=0.15
然后在目标框内部随机采样点：
local_x: -length/2 到 length/2
local_y: -width/2 到 width/2
local_z: -0.8 到 1.2
再根据 yaw 旋转，并平移到世界坐标：
x = obj_x + cos(yaw) * local_x - sin(yaw) * local_y
y = obj_y + sin(yaw) * local_x + cos(yaw) * local_y
最后给障碍物点设置较高反射强度：
intensity: 0.4~1.0
所以最终点云是：
背景点 + 车辆点 + 行人点 + 锥桶点
2. 点云怎么转 BEV
核心函数：
def points_to_bev(points, x_range=(0.0, 40.0), y_range=(-20.0, 20.0), resolution=0.1):
含义：
只看前方 0~40m
只看左右 -20~20m
每个像素代表 0.1m
因此 BEV 图大小是：
height = 40 / 0.1 = 400
width = 40 / 0.1 = 400
先过滤范围外的点：
mask = (
    (points[:, 0] >= x_min)
    & (points[:, 0] < x_max)
    & (points[:, 1] >= y_min)
    & (points[:, 1] < y_max)
)
pts = points[mask]
然后创建空 BEV 网格：
bev = np.zeros((height, width), dtype=np.float32)
再把真实坐标映射到图像坐标：
rows = ((pts[:, 0] - x_min) / resolution).astype(np.int32)
cols = ((pts[:, 1] - y_min) / resolution).astype(np.int32)
映射关系是：
x -> row
y -> col
例如点 [12.0, 3.0, 0.2, 0.8]：
row = (12.0 - 0.0) / 0.1 = 120
col = (3.0 - (-20.0)) / 0.1 = 230
然后把 intensity 填进对应网格：
np.maximum.at(bev, (rows, cols), pts[:, 3])
如果多个点落到同一个网格，就保留最大的 intensity。
最后转成图片灰度值：
return (bev * 255).clip(0, 255).astype(np.uint8)
3. 总结
当前 BEV 构建逻辑是：
1. 生成 [x, y, z, intensity] 点云
2. 过滤感兴趣区域
3. 按 0.1m 分辨率建立 400x400 网格
4. 将 x/y 坐标映射到 row/col
5. 每个网格保留最大 intensity
6. 转成 0~255 的灰度图片
一句话概括：
BEV 就是把 3D 点云投影到 x-y 平面，并离散成固定分辨率的二维栅格图。