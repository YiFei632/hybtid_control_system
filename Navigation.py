
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from astar import astar  # 假设A*算法在astar模块中

class Navigation:
    def __init__(self):
        # 读取环境地图
        file_path = 'map_binary.txt'
        with open(file_path, 'r') as file:
            map_data = file.readlines()
        map_matrix = []
        for line in map_data:
            line = line.strip()
            if line:
                row = list(map(int, line.split()))
                map_matrix.append(row)
        self.envmap = np.array(map_matrix)

    def smooth_path(self, path):
        """
        对路径进行平滑处理。
        :param path: 原始路径，格式为 [[x1, y1], [x2, y2], ...]
        :return: 平滑后的路径，格式为 [[x1, y1], [x2, y2], ...]
        """
        # 计算累积弦长作为参数
        dist = np.cumsum(np.sqrt(np.diff(path[:, 0]) ** 2 + np.diff(path[:, 1]) ** 2))
        dist = np.insert(dist, 0, 0)
        t_param = dist / dist[-1]

        # Cubic Spline插值
        cs_x = CubicSpline(t_param, path[:, 0])
        cs_y = CubicSpline(t_param, path[:, 1])

        # 假设速度v = 1单位/秒
        v = 1.0
        total_length = dist[-1]
        total_time = total_length / v

        # 生成时间序列，每隔0.1秒取一个点
        t_new = np.arange(0.0, total_time, 0.1)
        t_interp = t_new / total_time  # 将时间转换为路径参数

        # 计算平滑后的x和y
        x_smooth = cs_x(t_interp)
        y_smooth = cs_y(t_interp)

        # 返回平滑后的路径
        return np.column_stack((x_smooth, y_smooth))

    def add_angles_with_constraints(self, path, max_turn_rate):
        """
        为路径添加角度约束，并计算每个点的朝向角度。
        :param path: 平滑后的路径，格式为 [[x1, y1], [x2, y2], ...]
        :param max_turn_rate: 最大转向速率，单位为度/秒
        :return: 带有角度的路径，格式为 [Point(x, y, heading), ...]
        """
        # 计算导数，得到切线方向
        dx_dt = np.gradient(path[:, 0])
        dy_dt = np.gradient(path[:, 1])

        # 计算朝向角度
        angle_rad = np.arctan2(dy_dt, dx_dt)
        angle_deg = np.degrees(angle_rad)
        angle_deg = angle_deg % 360  # 确保角度在0到360度之间

        # 将路径和角度组合成Point对象
        class Point:
            def __init__(self, x, y, heading):
                self.x = x
                self.y = y
                self.heading = heading

        path_with_angles = [Point(x, y, heading) for x, y, heading in zip(path[:, 0], path[:, 1], angle_deg)]

        return path_with_angles

    def get_environment_map(self):
        """
        返回环境地图。
        :return: 环境地图，格式为 numpy 数组
        """
        return self.envmap

# 示例用法
if __name__ == '__main__':
    nav = Navigation()

    # 设置起点和终点
    start = [50, 421]
    goal = [350, 106]
    epsilon = 1.0
    fpath, cost, displaymap = astar(nav.get_environment_map(), start, goal, epsilon=1, inflate_factor=5)

    # 路径平滑处理
    smoothed_path = nav.smooth_path(fpath)

    # 计算角度并施加物理约束
    smoothed_path_with_angles = nav.add_angles_with_constraints(smoothed_path, max_turn_rate=10.0)

    # 输出结果
    for point in smoothed_path_with_angles:
        print(f"x: {point.x:.2f}, y: {point.y:.2f}, Angle: {point.heading:.2f} degrees")

    # 可选：绘制原始路径和平滑后的路径对比
    plt.figure()
    plt.imshow(displaymap, cmap='gray')
    plt.plot(fpath[:, 0], fpath[:, 1], 'ro-', label='Original Path')
    plt.plot(smoothed_path[:, 0], smoothed_path[:, 1], 'b.-', label='Smoothed Path')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()