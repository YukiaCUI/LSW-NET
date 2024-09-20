#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"

class PublishBaselinkScan
{
    private:
        ros::NodeHandle node;

        ros::Subscriber sub;
        ros::Publisher pub;

        int count = 0;
        bool sick = false; // 是否扩大角度分辨率
    
    public:
        PublishBaselinkScan(ros::NodeHandle n);
        ~PublishBaselinkScan() {};

        void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg);
};
