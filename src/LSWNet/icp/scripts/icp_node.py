#!/usr/bin/env python
# coding=utf-8 

import icp
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import tf2_ros
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
import math
import time

class ICP_Node() : 
    def __init__(self) : 
        self.sub = rospy.Subscriber('scan', LaserScan, self.laserScanCallBack)
        self.pub = rospy.Publisher('icp_odom', Odometry, queue_size=10)
        self.odom_broadcaster = TransformBroadcaster()
        self.odom = np.identity(3)
        self.first_frame = True

        self.save_poses = True
        self.file_name = "/home/shiwb/icp.txt"

        self.duration_times = []
        self.iteration_counts = []

    def printRuntime(self) : 
        print(np.mean(self.duration_times))
        print(np.mean(self.iteration_counts))

    def laserScanCallBack(self, msg) :
        # print("get new frame!")
        if self.first_frame : 
            self.convertScan2PointCloud(msg)
            self.first_frame = False
        else :
            self.A = self.B
            self.convertScan2PointCloud(msg)
            start_time = time.time()
            T, distances, iterations = icp.icp(self.B, self.A, tolerance=0.000001)
            end_time = time.time()
            print("icp_time: %f" % (end_time - start_time))
            self.duration_times.append(end_time - start_time)
            self.iteration_counts.append(iterations)
            x, y, theta = self.matrix_to_cartesian(T)
            # print("trans: %f,%f,%f: " % (x,y,theta))
            if(math.sqrt(x*x + y*y) < 0.3) : 
                self.odom = np.dot(self.odom, T)
            self.publish_odometry(msg)


    def convertScan2PointCloud(self, msg) :
        pc = []
        for i in range(len(msg.ranges)) :
            r = msg.ranges[i]
            if math.isnan(r) or math.isinf(r) : 
                continue
            if r > msg.range_min and r < msg.range_max : 
                angle = msg.angle_min + i * msg.angle_increment
                point = [r * math.cos(angle), r * math.sin(angle)]
                pc.append(point)
        self.B = np.array(pc)


    def matrix_to_cartesian(self, T):
        # 提取旋转矩阵
        R = T[:2, :2]
        # 提取平移向量
        p = T[:2, 2]
        # 计算欧拉角
        yaw = math.atan2(R[1, 0], R[0, 0])*180/np.pi
        # 返回笛卡尔坐标
        return p[0], p[1], yaw


    def publish_odometry(self, msg):
        # 更新里程计信息
        x, y, theta = self.matrix_to_cartesian(self.odom)
        print("odom: %f,%f,%f: " % (x,y,theta))

        # 发布TransformStamped消息
        # odom_trans = TransformStamped()
        # odom_trans.header.stamp = msg.header.stamp # rospy.Time.now()
        # odom_trans.header.frame_id = "odom"
        # odom_trans.child_frame_id = "base_link"
        # odom_trans.transform.translation.x = x
        # odom_trans.transform.translation.y = y
        
        # odom_trans.transform.rotation.w = math.cos(theta/180*math.pi / 2.0)
        # odom_trans.transform.rotation.z = math.sin(theta/180*math.pi / 2.0)
        # self.odom_broadcaster.sendTransform(odom_trans)
 
        # 发布Odometry消息
        odom_msg = Odometry()
        odom_msg.header.stamp = msg.header.stamp # rospy.Time.now()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"
        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.orientation.w = math.cos(theta/180*math.pi / 2.0)
        odom_msg.pose.pose.orientation.z = math.sin(theta/180*math.pi / 2.0) 
        self.pub.publish(odom_msg)

        # 保存位姿到文件
        if self.save_poses == True : 
            with open(self.file_name, 'a') as f : 
                f.write(str(msg.header.stamp.to_sec())+" "+str(x)+" "+str(y)+" "+str(theta/180*math.pi)+"\n")


if __name__ == "__main__" :
    rospy.init_node('icp_node', anonymous=True)
    node = ICP_Node()
    rospy.spin()
    node.printRuntime()
