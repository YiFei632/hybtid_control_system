#!/usr/bin/env python
# coding:utf-8
import rospy
import numpy as np
import scipy.sparse as sparse

import math
import tf
# from pyMPC.mpc import MPCController
from scipy.integrate import ode
from geometry_msgs.msg import Pose2D as pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry as odom
import tf.transformations

# 系统矩阵
""" A=np.zeros((3,3))
B=np.mat([[1,0],[0,0],[0,1]])

n_x=A.shape[0]
n_p=B.shape[1]

Ts=0.1 #sampling time """

x_ref=np.array([[0.0,0.0,0.0]]).T
x_fdb=np.array([[0.0,0.0,0.0]]).T
spd=np.array([[0.0,0.0]]).T
spd_ref=np.array([[0.0,0.0]]).T

""" xmin = np.array([-np.inf,-np.inf,-np.inf])
xmax = np.array([np.inf,np.inf,np.inf])

umin = np.array([-3.0,-3.0])
umax = np.array([3.0,3.0])

dumin = np.array([-np.inf,-np.inf])
dumax = np.array([np.inf,np.inf])

Qx = sparse.diags([1.0,1.0,1.0])
QxN = sparse.diags([1.0,1.0,1.0])
Qu = sparse.diags([0.1,0.1])
QDu = sparse.diags([0.1,0.1])

Np=10 """

# Kp = np.mat([[5,0,0],[0,0,5]])


def pose_callback(data):
    x_ref[0]=data.x
    x_ref[1]=data.y
    x_ref[2]=data.theta
    print(x_ref)

def odom_callback(data):
    x=data.pose.pose.position.x
    y=data.pose.pose.position.y
    (roll,pitch,yaw)=tf.transformations.euler_from_quaternion([data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w])
    x_fdb[0]=x
    x_fdb[1]=y
    x_fdb[2]=yaw
    #print(x_fdb)

def subscriber():
    rospy.Subscriber('pos_ref',pose,pose_callback)
    rospy.Subscriber('odom',odom,odom_callback)

def update():
    """ B[0,0]=math.cos(x_fdb[2])
    B[1,0]=math.sin(x_fdb[2]) """

    #Kp[0,0]=math.cos(x_fdb[2])
    #p[0,1]=math.sin(x_fdb[2])

""" def mpc():
    K = MPCController(A,B,Np=Np,x0=x_fdb,xref=x_ref,Qx=Qx,QxN=QxN,Qu=Qu,QDu=QDu,xmin=xmin,xmax=xmax,umin=umin,umax=umax,Dumin=dumin,Dumax=dumax)
    K.setup()
    spd = K.output()
    msg.linear.x=spd[0]
    msg.angular.z=spd[1]
    pub.publish(msg) """

def pid():
    """ spd = Kp * (x_ref - x_fdb)
    #print(spd) """
    x_err = x_ref[0]-x_fdb[0]
    y_err = x_ref[1]-x_fdb[1]
    print(x_err,'\t',y_err,'\t')
    #theta_err = x_ref[2]-x_fdb[2]

    if math.sqrt(x_err*x_err+y_err*y_err)>0.05:
        if x_ref[0]!=0.0:
            theta_ref=math.atan(x_ref[1]/x_ref[0])
        else:
            theta_ref=1.5708
        theta_err = theta_ref - x_fdb[2]
        if abs(theta_err) > 0.005:
            spd[0]=0.01
            spd[1]=10*theta_err
        else:
            spd[0]=2.5*(math.sqrt(x_err*x_err+y_err*y_err))
            spd[1]=0.01
    elif abs(x_ref[2]-x_fdb[2])>0.005:
        theta_err=x_ref[2]-x_fdb[2]
        spd[0]=0.01
        spd[1]=10*theta_err
    else:
        spd[0]=0.01
        spd[1]=0.01

    msg.linear.x=spd[0]
    msg.angular.z=spd[1]
    pub.publish(msg)
    #print(spd)



if __name__ == '__main__':
    # 创建节点
    rospy.init_node("limo_mpc")
    np.set_printoptions(precision=3)
    

    subscriber()
    pub=rospy.Publisher('cmd_vel',Twist,queue_size=10)
    rate=rospy.Rate(20)
    msg=Twist()

    while not rospy.is_shutdown():
        #subscriber()
        # update()
        # mpc()
        pid()
        rate.sleep()
    #创建ref的subscriber
    # rospy.Subscriber('position',pose,pose_callback)
    