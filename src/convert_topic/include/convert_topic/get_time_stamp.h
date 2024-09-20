#include <iostream>
#include <fstream>
#include <string>
#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"

class GetTimeStamp
{
    private:
        ros::NodeHandle node;

        ros::Subscriber sub;

    
    public:
       GetTimeStamp(ros::NodeHandle n);
        ~GetTimeStamp() {};

        void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg);
};
