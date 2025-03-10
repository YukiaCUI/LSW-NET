#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan
import torch               #用于构建神经网络，torch是PyTorch的核心库，提供了张量操作、自动求导等功能

def laser_scan_callback(msg):
    data = {
        "header": {
            "seq": msg.header.seq,
            "stamp": {
                "secs": msg.header.stamp.secs,
                "nsecs": msg.header.stamp.nsecs
            },
            "frame_id": msg.header.frame_id
        },
        "angle_min": msg.angle_min,
        "angle_max": msg.angle_max,
        "angle_increment": msg.angle_increment,
        "time_increment": msg.time_increment,
        "scan_time": msg.scan_time,
        "range_min": msg.range_min,
        "range_max": msg.range_max,
        "ranges": msg.ranges,
        "intensities": msg.intensities
    } 
    X = msg.ranges
    # X = torch.tensor(X, dtype=torch.float32)
    path = "/home/rosnoetic/pointcloud/src/point_cloud_processing/data/laser_scan.txt"   
    file = open(path,"w")
    file.write(str(X))
    file.close()
    rospy.loginfo("I heard:%s",X)

if __name__ == "__main__":
    #2.初始化 ROS 节点:命名(唯一)
    rospy.init_node("listener_pp")
    #3.实例化 订阅者 对象
    sub = rospy.Subscriber("scan_processed",LaserScan,laser_scan_callback)
    #4.处理订阅的消息(回调函数)
    #5.设置循环调用回调函数
    rospy.spin()
