import rospy
from geometry_msgs.msg import Pose2D, PoseStamped
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import UInt8MultiArray
import numpy as np
from astar import astar
import tf.transformations
import math  # 导入math模块

class PathPlannerNode:
    def __init__(self):
        rospy.init_node('path_planner_node', anonymous=True)

        # 订阅当前位姿
        self.pose_sub = rospy.Subscriber('odom', Odometry, self.odom_callback)

        # 订阅目标点（单个goal）
        self.goal_sub = rospy.Subscriber('/goal', Pose2D, self.goal_callback)
        self.goal = None

        # 订阅 man 位置（单个man目标，PoseStamped类型）
        self.man_sub = rospy.Subscriber('/man', PoseStamped, self.man_callback)
        self.man_target = None

        # 订阅地图
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.map_data = None
        self.map_info = None  # 用于存储地图的元数据

        # 发布路径
        self.pose_pub = rospy.Publisher('pos_ref', Pose2D, queue_size=10)

        # 定时调用路径规划函数
        self.timer = rospy.Timer(rospy.Duration(1.0), self.timer_callback)

        self.current_pose = None

    def odom_callback(self, data):
        self.current_pose = Pose2D()
        self.current_pose.x = data.pose.pose.position.x
        self.current_pose.y = data.pose.pose.position.y
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
            [data.pose.pose.orientation.x, data.pose.pose.orientation.y,
             data.pose.pose.orientation.z, data.pose.pose.orientation.w])
        self.current_pose.theta = yaw

    def timer_callback(self, event):
        self.plan_path()

    def goal_callback(self, msg):
        self.goal = msg

    def man_callback(self, msg):
        self.man_target = msg

    def map_callback(self, msg):
        # 将 OccupancyGrid 转换为二维数组
        width = msg.info.width
        height = msg.info.height
        data = msg.data
        self.map_data = np.array(data).reshape(height, width)
        self.map_info = msg.info  # 保存地图的元数据，如origin和resolution

    def plan_path(self):
        if self.current_pose is None or self.map_data is None:
            return

        # 优先导航到man目标，如果man目标为空，则导航到goal目标
        if self.man_target and self.man_target.pose.position:
            target = self.man_target.pose.position
            target_x = target.x
            target_y = target.y
        elif self.goal:
            target = self.goal
            target_x = target.x
            target.y = target.y
        else:
            rospy.loginfo("No goals or man positions available.")
            return

        # 提取起点和终点坐标
        start = [self.current_pose.x, self.current_pose.y]
        goal = [target_x, target_y]

        # 转换起点和终点到地图的索引
        map_origin_x = self.map_info.origin.position.x
        map_origin_y = self.map_info.origin.position.y
        map_resolution = self.map_info.resolution

        start_idx = [
            int((start[0] - map_origin_x) / map_resolution),
            int((start[1] - map_origin_y) / map_resolution)
        ]
        goal_idx = [
            int((goal[0] - map_origin_x) / map_resolution),
            int((goal[1] - map_origin_y) / map_resolution)
        ]

        # 调用 A* 算法规划路径
        fpath, cost, displaymap = astar(self.map_data, start_idx, goal_idx, epsilon=1.0, inflate_factor=5)

        if fpath is None:
            rospy.logwarn("No path found.")
            return

        # 创建 Path 消息
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"

        # 发布路径点
        for i in range(len(fpath)):
            point = fpath[i]
            next_point = fpath[i+1]
            dx = next_point[0] - point[0]
            dy = next_point[1] - point[1]
            theta = math.atan2(dy, dx)
            pose_2d = Pose2D()
            pose_2d.x = point[0] * map_resolution + map_origin_x
            pose_2d.y = point[1] * map_resolution + map_origin_y
            pose_2d.theta = theta
            self.pose_pub.publish(pose_2d)

        # 处理最后一个点

if __name__ == '__main__':
    try:
        planner = PathPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
