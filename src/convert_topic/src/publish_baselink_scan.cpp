#include "convert_topic/publish_baselink_scan.h"

PublishBaselinkScan::PublishBaselinkScan(ros::NodeHandle n) : node(n)
{
    sub = node.subscribe("/scan", 1, &PublishBaselinkScan::scanCallback, this);
    pub = node.advertise<sensor_msgs::LaserScan>("/scan2", 100);

    count = 0;

    n.param("/publish_baselink_scan/sick", sick, false);
}

void PublishBaselinkScan::scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
{
    count ++;

    sensor_msgs::LaserScan new_msg;

    new_msg.header = msg->header;
    new_msg.angle_min = msg->angle_min;
    new_msg.angle_max = msg->angle_max;
    
    new_msg.time_increment = msg->time_increment;
    new_msg.scan_time = msg->scan_time;
    new_msg.range_min = msg->range_min;
    new_msg.range_max = 15.0;
    if(sick)
    {
        new_msg.angle_increment = msg->angle_increment*5;
        for(int i=0; i<msg->ranges.size(); i++)
        {
            if(i%5==0)
            {
                if(msg->ranges[i]>9.0)
                    new_msg.ranges.emplace_back(INFINITY);
                else
                    new_msg.ranges.emplace_back(msg->ranges[i]);
                new_msg.intensities.emplace_back(msg->intensities[i]);
            }
        }
    }
    else
    {
        new_msg.angle_increment = msg->angle_increment;
        new_msg.ranges = msg->ranges;
        new_msg.intensities = msg->intensities;
    }
    
    

    new_msg.header.frame_id = "base_link";
    // new_msg.header.frame_id = "laser_test";
    ROS_INFO("publish baselink scan complete, %d !", count);
    pub.publish(new_msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "publish_baselink_scan");
    ros::NodeHandle n;
    PublishBaselinkScan object(n);
    ros::spin();
    return 0;
}
