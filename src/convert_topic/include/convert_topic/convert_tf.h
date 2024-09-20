#include "ros/ros.h"
#include "tf/tfMessage.h"

class ConvertTf
{
    private:
        ros::NodeHandle node;

        ros::Subscriber sub;
        ros::Publisher pub;
    
    public:
        ConvertTf(ros::NodeHandle n);
        ~ConvertTf() {};

        void tfCallback(const tf::tfMessage::ConstPtr& msg);
};
