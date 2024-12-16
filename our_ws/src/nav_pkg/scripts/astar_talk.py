import rospy
from geometry_msgs.msg import Pose2D, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
from astar import astar
import tf.transformations
import math

class PathPlannerNode:
    def __init__(self):
        rospy.init_node('path_planner_node', anonymous=True)

        # Subscribe to current pose
        self.pose_sub = rospy.Subscriber('odom', Odometry, self.odom_callback)

        # Subscribe to single goal (grid coordinates)
        self.goal_sub = rospy.Subscriber('/goal', Pose2D, self.goal_callback)
        self.goal = None

        # Subscribe to single man target (PoseStamped type, in meters)
        self.man_sub = rospy.Subscriber('/man', PoseStamped, self.man_callback)
        self.man_target = None

        # Subscribe to map
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.map_data = None
        self.map_info = None  # To store map metadata

        # Publish path
        self.pose_pub = rospy.Publisher('pos_ref', Pose2D, queue_size=10)

        # Timer to call path planning function
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
        self.goal = msg  # Goal is in grid coordinates

    def man_callback(self, msg):
        if msg.pose.position:
            man_x = msg.pose.position.x
            man_y = msg.pose.position.y
            # Convert man position to grid indices
            map_origin_x = self.map_info.origin.position.x
            map_origin_y = self.map_info.origin.position.y
            map_resolution = self.map_info.resolution
            man_idx_x = int((man_x - map_origin_x) / map_resolution)
            man_idx_y = int((man_y - map_origin_y) / map_resolution)
            self.man_target = [man_idx_x, man_idx_y]
        else:
            self.man_target = None

    def map_callback(self, msg):
        # Convert OccupancyGrid to 2D numpy array
        width = msg.info.width
        height = msg.info.height
        data = msg.data
        self.map_data = np.array(data).reshape(height, width)
        self.map_info = msg.info  # Store map metadata

    def plan_path(self):
        if self.current_pose is None or self.map_data is None:
            return

        # Prioritize man_target, if available, else use goal
        if self.man_target:
            goal_idx = self.man_target
        elif self.goal:
            goal_idx = [self.goal.x, self.goal.y]
        else:
            rospy.loginfo("No goals or man positions available.")
            return

        # Convert current pose to grid indices
        map_origin_x = self.map_info.origin.position.x
        map_origin_y = self.map_info.origin.position.y
        map_resolution = self.map_info.resolution
        start_idx = [
            int((self.current_pose.x - map_origin_x) / map_resolution),
            int((self.current_pose.y - map_origin_y) / map_resolution)
        ]

        # Call A* algorithm for path planning
        fpath, cost, displaymap = astar(self.map_data, start_idx, goal_idx, epsilon=1.0, inflate_factor=5)

        if fpath is None:
            rospy.logwarn("No path found.")
            return

        # Publish path points
        for i in range(len(fpath)-1):
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

        # Handle last point
        if len(fpath) > 0:
            last_point = fpath[-1]
            pose_2d = Pose2D()
            pose_2d.x = last_point[0] * map_resolution + map_origin_x
            pose_2d.y = last_point[1] * map_resolution + map_origin_y
            pose_2d.theta = 0.0
            self.pose_pub.publish(pose_2d)

if __name__ == '__main__':
    try:
        planner = PathPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
