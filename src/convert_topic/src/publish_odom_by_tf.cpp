#include "convert_topic/publish_odom_by_tf.h"

ConvertTfToTopic::ConvertTfToTopic(ros::NodeHandle n) : node(n)
{
    sub = node.subscribe("/tf", 1, &ConvertTfToTopic::tfCallback, this);
    pub = node.advertise<nav_msgs::Odometry>("/odom", 1);
}

void ConvertTfToTopic::tfCallback(const tf::tfMessage::ConstPtr& msg)
{
    nav_msgs::Odometry odom_msg;

    int size = msg->transforms.size();
    for(int i=0; i<size; i++)
    {
        if(msg->transforms[i].header.frame_id == "odom")
        {
            odom_msg.header = msg->transforms[i].header;
            odom_msg.child_frame_id = msg->transforms[i].child_frame_id;
            odom_msg.pose.pose.position.x = msg->transforms[i].transform.translation.x;
            odom_msg.pose.pose.position.y = msg->transforms[i].transform.translation.y;
            odom_msg.pose.pose.position.z = msg->transforms[i].transform.translation.z;
            odom_msg.pose.pose.orientation = msg->transforms[i].transform.rotation;

            pub.publish(odom_msg);
            ROS_INFO("publish new tf topic complete!"); 
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "convert_tf");
    ros::NodeHandle n;
    ConvertTfToTopic object(n);
    ros::spin();
    return 0;
}