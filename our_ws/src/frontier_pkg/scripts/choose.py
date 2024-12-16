import rospy
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from geometry_msgs.msg import Pose2D, Odometry
from nav_msgs.msg import OccupancyGrid

class GoalNavigator:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('goal_navigator', anonymous=True)
        
        # 发布目标点
        self.goal_pub = rospy.Publisher('/goal_point', Pose2D, queue_size=1)
        
        # 订阅栅格地图
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        
        # 订阅小车的里程计数据
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # 存储地图数据
        self.occupancy_grid = None
        self.map_info = None  # 存储地图的元数据
        
        # 存储小车的位置
        self.robot_position = None
    
    def map_callback(self, msg):
        # 将OccupancyGrid消息转换为numpy数组
        self.occupancy_grid = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_info = msg.info  # 存储地图的元数据
    
    def odom_callback(self, msg):
        # 获取小车的位置信息（世界坐标系）
        self.robot_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
    
    def grid_to_world(self, grid_point):
        """
        将栅格坐标转换为世界坐标
        :param grid_point: 栅格坐标 (行, 列)
        :return: 世界坐标 (x, y)
        """
        if self.map_info is None:
            rospy.logwarn("地图元数据不可用。")
            return None
        
        row, col = grid_point
        x = self.map_info.origin.position.x + (col + 0.5) * self.map_info.resolution
        y = self.map_info.origin.position.y + (row + 0.5) * self.map_info.resolution
        return (x, y)
    
    def find_nearest_edge_point(self, robot_position, safety_distance):
        if self.occupancy_grid is None or self.map_info is None:
            rospy.logwarn("等待地图数据...")
            return None
        
        # 将占据栅格地图转换为二值图像
        binary_map = np.where(self.occupancy_grid == 100, 255, 0).astype(np.uint8)  # 障碍物设为255
        
        # 检测边缘
        edges = cv2.Canny(binary_map, 50, 150)
        
        # 提取边缘点的坐标（栅格坐标）
        edge_points = np.column_stack(np.where(edges > 0))
        
        # 如果没有检测到任何边缘点
        if edge_points.size == 0:
            raise ValueError("地图中未检测到边缘点。")
        
        # 使用DBSCAN对边缘点进行聚类
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
            raise ValueError("聚类后未找到候选边缘点。")
        
        # 将候选边缘点的栅格坐标转换为世界坐标
        world_candidate_points = [self.grid_to_world(point) for point in candidate_points]
        
        # 找到距离小车最近的候选边缘点，并确保满足安全距离
        robot_position = np.array(robot_position)
        world_candidate_points = np.array(world_candidate_points)
        distances = np.linalg.norm(world_candidate_points - robot_position, axis=1)
        
        # 过滤掉不满足安全距离的点
        valid_points = world_candidate_points[distances >= safety_distance]
        
        # 如果没有满足安全距离的点
        if len(valid_points) == 0:
            raise ValueError("未找到满足安全距离的边缘点。")
        
        # 找到距离小车最近的点
        nearest_point_index = np.argmin(distances)
        nearest_point = world_candidate_points[nearest_point_index]
        
        return nearest_point  # 返回世界坐标 (x, y)
    
    def publish_goal(self, goal):
        goal_msg = Pose2D()
        goal_msg.x = goal[0]  # 世界坐标 (x, y)
        goal_msg.y = goal[1]
        goal_msg.theta = 0  # 目标点的角度设为0（可根据需要修改）
        self.goal_pub.publish(goal_msg)
    
    def navigate(self, safety_distance):
        if self.robot_position is None:
            rospy.logwarn("等待小车位置数据...")
            return
        
        try:
            nearest_point = self.find_nearest_edge_point(self.robot_position, safety_distance)
            if nearest_point:
                rospy.loginfo(f"发布目标点: {nearest_point}")
                self.publish_goal(nearest_point)
        except ValueError as e:
            rospy.logwarn(f"错误: {e}")

if __name__ == '__main__':
    navigator = GoalNavigator()
    
    safety_distance = 0.1  # 安全距离，可根据需要调整
    
    rate = rospy.Rate(1)  # 1Hz
    while not rospy.is_shutdown():
        navigator.navigate(safety_distance)
        rate.sleep()
