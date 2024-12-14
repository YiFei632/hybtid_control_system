#!/usr/bin/env python


import rospy

from sensor_msgs.msg import Image

from ultralytics import YOLO

import ros_numpy

rospy.init_node("ultralytics")
rospy.loginfo("Initialized ultralytics node")

detection_model = YOLO("/home/lql/lql_ws/src/reco_pkg/scripts/yolo11n.pt")
rospy.loginfo("Loaded YOLO model")
rospy.sleep(rospy.Duration(1))
det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)

def callback(data):
    """Callback function to process image and publish annotated images."""
    array = ros_numpy.numpify(data)
    rospy.loginfo("into callback function")
    if det_image_pub.get_num_connections():
        det_result = detection_model(array)
        det_annotated = det_result[0].plot(show=False)
        det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))
        rospy.loginfo("pub a detected result")



rospy.Subscriber("/myImage", Image, callback)
rospy.loginfo("Subscribed to /myImage")
while not rospy.is_shutdown():
    rospy.spin()