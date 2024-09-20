#include "ros/ros.h"
#include "sensor_msgs/Imu.h"

class ConvertImu
{
    private:
        ros::NodeHandle node;

        ros::Subscriber sub;
        ros::Publisher pub;
    
    public:
        ConvertImu(ros::NodeHandle n);
        ~ConvertImu(){};

        void imuCallback(const sensor_msgs::Imu::ConstPtr& msg);
};
