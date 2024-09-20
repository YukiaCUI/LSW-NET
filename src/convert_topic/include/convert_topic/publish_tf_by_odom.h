#include "ros/ros.h"
#include "tf/tfMessage.h"
#include <tf/transform_broadcaster.h>
#include "nav_msgs/Odometry.h"
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>


class ConvertOdomToTf
{
    private:
        ros::NodeHandle node;

        ros::Subscriber sub;
    
    public:
        ConvertOdomToTf(ros::NodeHandle n);
        ~ConvertOdomToTf() {};

        void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
};