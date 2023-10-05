"""# 将h×w×3数组转换为N×3的数组
def hw3ToN3(points):
    height, width = points.shape[0:2]

    points_1 = points[:, :, 0].reshape(height * width, 1)
    points_2 = points[:, :, 1].reshape(height * width, 1)
    points_3 = points[:, :, 2].reshape(height * width, 1)

    points_ = np.hstack((points_1, points_2, points_3))

    return points_"""


"""# 深度、颜色转换为点云
def DepthColor2Cloud(points_3d, colors):
    rows, cols = points_3d.shape[0:2]
    size = rows * cols

    points_ = hw3ToN3(points_3d)
    colors_ = hw3ToN3(colors).astype(np.int64)

    # 颜色信息
    blue = colors_[:, 0].reshape(size, 1)
    green = colors_[:, 1].reshape(size, 1)
    red = colors_[:, 2].reshape(size, 1)

    rgb = np.left_shift(blue, 0) + np.left_shift(green, 8) + np.left_shift(red, 16)

    # 将坐标+颜色叠加为点云数组
    pointcloud = np.hstack((points_, rgb)).astype(np.float32)

    # 删掉一些不合适的点
    X = pointcloud[:, 0]
    Y = pointcloud[:, 1]
    Z = pointcloud[:, 2]

    # 下面参数是经验性取值，需要根据实际情况调整
    remove_idx1 = np.where(Z <= 0)
    remove_idx2 = np.where(Z > 15000)
    remove_idx3 = np.where(X > 10000)
    remove_idx4 = np.where(X < -10000)
    remove_idx5 = np.where(Y > 10000)
    remove_idx6 = np.where(Y < -10000)
    remove_idx = np.hstack(
        (remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0], remove_idx5[0], remove_idx6[0]))

    pointcloud_1 = np.delete(pointcloud, remove_idx, 0)

    return pointcloud_1


# 点云显示
def view_cloud(pointcloud):
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_array(pointcloud)

    try:
        visual = pcl.pcl_visualization.CloudViewing()
        visual.ShowColorACloud(cloud)
        v = True
        while v:
            v = not (visual.WasStopped())
    except:
        pass"""


"""def getDepthMapWithQ(disparityMap: np.ndarray, Q: np.ndarray) -> np.ndarray:
    points_3d = cv2.reprojectImageTo3D(disparityMap, Q)
    depthMap = points_3d[:, :, 2]
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0

    return depthMap.astype(np.float32)"""

"""
# 使用open3d库绘制点云
colorImage = o3d.geometry.Image(iml)
depthImage = o3d.geometry.Image(depthMap)
rgbdImage = o3d.geometry.RGBDImage().create_from_color_and_depth(colorImage, depthImage, depth_scale=1000.0,
                                                                 depth_trunc=np.inf)
intrinsics = o3d.camera.PinholeCameraIntrinsic()
# fx = Q[2, 3]
# fy = Q[2, 3]
# cx = Q[0, 3]
# cy = Q[1, 3]
fx = config.cam_matrix_left[0, 0]
fy = fx
cx = config.cam_matrix_left[0, 2]
cy = config.cam_matrix_left[1, 2]
print(fx, fy, cx, cy)
intrinsics.set_intrinsics(width, height, fx=fx, fy=fy, cx=cx, cy=cy)
extrinsics = np.array([[1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]])
pointcloud = o3d.geometry.PointCloud().create_from_rgbd_image(rgbdImage, intrinsic=intrinsics, extrinsic=extrinsics)
o3d.io.write_point_cloud("PointCloud.pcd", pointcloud=pointcloud)
o3d.visualization.draw_geometries([pointcloud], width=720, height=480)
sys.exit(0)

# 计算像素点的3D坐标（左相机坐标系下）
points_3d = cv2.reprojectImageTo3D(disp, Q)  # 参数中的Q就是由getRectifyTransform()函数得到的重投影矩阵

# 构建点云--Point_XYZRGBA格式
pointcloud = DepthColor2Cloud(points_3d, iml)

# 显示点云
view_cloud(points_3d)"""