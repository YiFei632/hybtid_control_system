import rospy
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid

#rospy geometry_msgs nav_msgs
class GoalNavigator:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('goal_navigator', anonymous=True)
        
        # 发布目标点队列
        self.goal_pub = rospy.Publisher('/goal_queue', Pose2D, queue_size=10)
        
        # 订阅栅格地图
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        
        # 存储地图
        self.occupancy_grid = None
        
        # 初始化目标队列
        self.goal_queue = []
        
    def map_callback(self, msg):
        # 将OccupancyGrid消息转换为numpy数组
        self.occupancy_grid = np.array(msg.data).reshape(msg.info.height, msg.info.width)
    
    def find_nearest_edge_point(self, robot_position, safety_distance):
        if self.occupancy_grid is None:
            rospy.logwarn("Waiting for map data...")
            return None
        
        # 将 gmapping 格式的占据栅格地图转换为二值图像
        binary_map = np.where(self.occupancy_grid == 100, 255, 0).astype(np.uint8)  # 将占据部分设为255（障碍物）
        
        # 检测边缘
        edges = cv2.Canny(binary_map, 50, 150)

        # 提取边缘点的坐标
        edge_points = np.column_stack(np.where(edges > 0))

        # 如果没有检测到任何边缘点
        if edge_points.size == 0:
            raise ValueError("No edge points detected in the map.")

        # 使用DBSCAN对边缘点聚类
        clustering = DBSCAN(eps=5, min_samples=10).fit(edge_points)
        labels = clustering.labels_

        # 选择聚类中心作为候选边缘点
        candidate_points = []
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  # 忽略噪声点
            cluster_points = edge_points[labels == label]
            cluster_center = np.mean(cluster_points, axis=0).astype(int)
            candidate_points.append(cluster_center)

        # 如果没有候选边缘点
        if len(candidate_points) == 0:
            raise ValueError("No candidate edge points found after clustering.")

        # 找到距离小车最近的候选边缘点，并确保满足安全距离
        robot_position = np.array(robot_position)
        candidate_points = np.array(candidate_points)
        valid_points = []

        for point in candidate_points:
            distances_to_obstacles = np.linalg.norm(edge_points - point, axis=1)
            if np.all(distances_to_obstacles >= safety_distance):
                valid_points.append(point)

        # 如果没有满足安全距离的点
        if not valid_points:
            raise ValueError("No valid edge point found within safety constraints.")

        valid_points = np.array(valid_points)
        distances = np.linalg.norm(valid_points - robot_position, axis=1)
        nearest_point_index = np.argmin(distances)
        nearest_point = valid_points[nearest_point_index]

        # 输出点的格式转换为A星算法所需的格式
        # 假设A星算法使用 (x, y) 格式
        return tuple(nearest_point)  # 返回为 (row, col) 格式

    def publish_goal(self, goal):
        goal_msg = Pose2D()
        goal_msg.x = goal[1]  # 假设目标点为 (row, col)，转换为 (x, y)
        goal_msg.y = goal[0]
        goal_msg.theta = 0  # 设定目标点的角度为0（可根据需求修改）
        self.goal_pub.publish(goal_msg)

    def navigate(self, robot_position, safety_distance):
        try:
            nearest_point = self.find_nearest_edge_point(robot_position, safety_distance)
            if nearest_point:
                rospy.loginfo(f"Publishing goal: {nearest_point}")
                self.publish_goal(nearest_point)
        except ValueError as e:
            rospy.logwarn(f"Error: {e}")

if __name__ == '__main__':
    navigator = GoalNavigator()

    # 假设机器人当前位置为 (50, 50)，安全距离为 5
    robot_position = (50, 50)
    safety_distance = 5

    rate = rospy.Rate(1)  # 1Hz
    while not rospy.is_shutdown():
        navigator.navigate(robot_position, safety_distance)
        rate.sleep()
