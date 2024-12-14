#!/usr/bin/env python

import rospy
import cv2
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def main():
    rospy.init_node('image_publisher', anonymous=True)
    image_pub = rospy.Publisher('/myImage', Image, queue_size=10)
    bridge = CvBridge()
    rate = rospy.Rate(0.5)  # 0.5 Hz

    # Path to the folder containing the images
    img_folder = '/home/lql/lql_ws/src/reco_pkg/test_img'
    img_files = sorted([f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    rospy.loginfo(f"Found {len(img_files)} images in {img_folder}")
    while not rospy.is_shutdown():
        for img_file in img_files:
            img_path = os.path.join(img_folder, img_file)
            frame = cv2.imread(img_path)
            #show 
            cv2.imshow('image', frame)
            if frame is None:
                rospy.logerr(f"Failed to read image from {img_path}")
                continue

            try:
                image_msg = bridge.cv2_to_imgmsg(frame, "bgr8")
                image_pub.publish(image_msg)
                rospy.loginfo(f"Published {img_file}")
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")

            rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass