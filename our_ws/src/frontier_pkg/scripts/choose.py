#!/usr/bin/env python
# coding:utf-8
import rospy
import numpy as np
from sklearn.cluster import DBSCAN
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from math import sqrt

class GoalNavigator:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('goal_navigator', anonymous=True)
        
        # 发布目标点
        self.goal_pub = rospy.Publisher('/goal_point', PointStamped, queue_size=1)  # 设置队列大小为1
        
        # 订阅栅格地图
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        
        # 订阅小车的里程计数据
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # 存储地图数据
        self.occupancy_grid = None
        self.map_info = None  # 存储地图的元数据
        
        # 存储小车的位置
        self.robot_position = None
        self.found = False
    
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
            rospy.logwarn("Map info is not available.")
            return None
        
        row, col = grid_point
        x = self.map_info.origin.position.x + (col + 0.5) * self.map_info.resolution
        y = self.map_info.origin.position.y + (row + 0.5) * self.map_info.resolution
        return (x, y)
    
    def find_nearest_edge_point(self, robot_position, safety_distance):
        if self.occupancy_grid is None or self.map_info is None:
            rospy.logwarn("Waiting for map data...")
            return None

        # 获取边缘点：遍历已探索区域（值为0），检查其周围是否有未探索区域（值为-1）
        edge_points = []
        rows, cols = self.occupancy_grid.shape
        for i in range(rows):
            for j in range(cols):
                if self.occupancy_grid[i, j] == 0:  # 已探索区域
                    # 检查四个邻域（上、下、左、右）是否是未探索区域（-1）
                    neighbors = [
                        (i-1, j), (i+1, j), (i, j-1), (i, j+1)  # 上、下、左、右邻居
                    ]
                    for ni, nj in neighbors:
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if self.occupancy_grid[ni, nj] == -1:  # 邻域是未探索区域
                                edge_points.append((i, j))
                                break

        # 如果没有检测到任何边缘点
        if len(edge_points) == 0:
            rospy.loginfo("No edge points detected in the map.")
            return None

        # 使用DBSCAN对边缘点进行聚类
        edge_points = np.array(edge_points)
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
            rospy.loginfo("No candidate edge points found after clustering.")
            return None

        # 将候选边缘点的栅格坐标转换为世界坐标
        world_candidate_points = [self.grid_to_world(point) for point in candidate_points]

        # 找到距离小车最近的候选边缘点
        robot_position = np.array(robot_position)
        world_candidate_points = np.array(world_candidate_points)
        distances = np.linalg.norm(world_candidate_points - robot_position, axis=1)

        # 找到距离小车最近的点
        nearest_point_index = np.argmin(distances)
        nearest_point = world_candidate_points[nearest_point_index]

        # 计算目标点：确保目标点与边缘点之间有安全距离
        goal_point = self.calculate_goal_point(robot_position, nearest_point, safety_distance)

        return goal_point  # 返回目标点
    
    def calculate_goal_point(self, robot_position, nearest_edge_point, safety_distance):
        """
        计算目标点，使得目标点与最近边缘点之间的距离等于安全距离，并且目标点位于小车和最近边缘点之间
        """
        robot_position = np.array(robot_position)
        nearest_edge_point = np.array(nearest_edge_point)
        
        # 计算从机器人到最近边缘点的向量
        direction_vector = nearest_edge_point - robot_position
        direction_vector_norm = direction_vector / np.linalg.norm(direction_vector)
        
        # 计算目标点
        goal_point = nearest_edge_point - direction_vector_norm * safety_distance
        
        return goal_point
    
    def publish_goal(self, goal):
        goal_msg = PointStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.point.x = goal[0]  # 世界坐标 (x, y)
        goal_msg.point.y = goal[1]
        goal_msg.point.z = 0  # 目标点的角度设为0（可根据需要修改）
        self.goal_pub.publish(goal_msg)
    
    def navigate(self, safety_distance):
        if self.robot_position is None:
            rospy.logwarn("Waiting for robot position data...")
            return
        
        try:
            goal_point = self.find_nearest_edge_point(self.robot_position, safety_distance)
            if goal_point is None:
                rospy.loginfo("No edge points found.")
                return
            
            rospy.loginfo(f"Publishing goal: {goal_point}")
            self.publish_goal(goal_point)
        except ValueError as e:
            rospy.logwarn(f"Error: {e}")

if __name__ == '__main__':
    navigator = GoalNavigator()
    
    safety_distance = 0.2  # 安全距离，可根据需要调整
    
    rate = rospy.Rate(1)  # 1Hz
    while not rospy.is_shutdown():
        navigator.navigate(safety_distance)
        rate.sleep()
