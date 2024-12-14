import rospy
from geometry_msgs.msg import Pose2D, PoseStamped
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import UInt8MultiArray
import numpy as np
from astar import astar

class PathPlannerNode:
    def __init__(self):
        rospy.init_node('path_planner_node', anonymous=True)

        # 订阅当前位姿
        self.pose_sub = rospy.Subscriber('/current_pose', Pose2D, self.pose_callback)

        # 订阅目标点队列
        self.goal_queue = []
        self.goal_sub = rospy.Subscriber('/goal_queue', Pose2D, self.goal_queue_callback)

        # 订阅 man 位置队列
        self.man_queue = []
        self.man_sub = rospy.Subscriber('/man_queue', Pose2D, self.man_queue_callback)

        # 订阅地图
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)

        # 发布路径
        self.path_pub = rospy.Publisher('/smoothed_path', Path, queue_size=10)

        self.current_pose = None
        self.map_data = None

    def pose_callback(self, msg):
        self.current_pose = msg

    def goal_queue_callback(self, msg):
        self.goal_queue.append(msg)

    def man_queue_callback(self, msg):
        self.man_queue.append(msg)

    def map_callback(self, msg):
        # 将 OccupancyGrid 转换为二维数组
        width = msg.info.width
        height = msg.info.height
        data = msg.data
        self.map_data = np.array(data).reshape(height, width)

    def plan_path(self):
        if self.current_pose is None or self.map_data is None:
            return

        if self.man_queue:
            # 优先导航到最近的 man 位置
            target = self.get_nearest_point(self.current_pose, self.man_queue)
        else:
            # 导航到最近的 goal 位置
            if self.goal_queue:
                target = self.get_nearest_point(self.current_pose, self.goal_queue)
            else:
                rospy.loginfo("No goals or man positions available.")
                return

        # 提取起点和终点坐标
        start = [self.current_pose.x, self.current_pose.y]
        goal = [target.x, target.y]

        # 调用 A* 算法规划路径
        fpath, cost, displaymap = astar(self.map_data, start, goal, epsilon=1.0, inflate_factor=5)

        if fpath is None:
            rospy.logwarn("No path found.")
            return

        # 创建 Path 消息
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"

        for point in fpath:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.orientation.w = 1.0  # 假设没有朝向
            path_msg.poses.append(pose)

        # 发布路径
        self.path_pub.publish(path_msg)

    def get_nearest_point(self, current_pose, queue):
        current_x = current_pose.x
        current_y = current_pose.y
        nearest = None
        min_dist = float('inf')

        for point in queue:
            dist = np.hypot(point.x - current_x, point.y - current_y)
            if dist < min_dist:
                min_dist = dist
                nearest = point

        return nearest

if __name__ == '__main__':
    try:
        planner = PathPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass