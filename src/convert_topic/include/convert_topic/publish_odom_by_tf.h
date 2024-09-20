#include "ros/ros.h"
#include "tf/tfMessage.h"
#include "nav_msgs/Odometry.h"

class ConvertTfToTopic
{
    private:
        ros::NodeHandle node;

        ros::Subscriber sub;
        ros::Publisher pub;
    
    public:
        ConvertTfToTopic(ros::NodeHandle n);
        ~ConvertTfToTopic() {};

        void tfCallback(const tf::tfMessage::ConstPtr& msg);
};
