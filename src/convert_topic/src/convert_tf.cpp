#include "convert_topic/convert_tf.h"

ConvertTf::ConvertTf(ros::NodeHandle n) : node(n)
{
    sub = node.subscribe("/tf3", 1, &ConvertTf::tfCallback, this);
    pub = node.advertise<tf::tfMessage>("/tf", 1);
}

void ConvertTf::tfCallback(const tf::tfMessage::ConstPtr& msg)
{
    tf::tfMessage tf_msg;

    int size = msg->transforms.size();
    for(int i=0; i<size; i++)
    {
        if(msg->transforms[i].header.frame_id == "base_link")
        {
            tf_msg.transforms.emplace_back(msg->transforms[i]);
        }
    }

    if(tf_msg.transforms.size() != 0)
    {
        pub.publish(tf_msg);
        ROS_INFO("publish new tf topic complete!");
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "convert_tf");
    ros::NodeHandle n;
    ConvertTf tfObject(n);
    ros::spin();
    return 0;
}