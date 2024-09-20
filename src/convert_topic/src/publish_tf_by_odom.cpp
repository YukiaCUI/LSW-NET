#include "convert_topic/publish_tf_by_odom.h"

ConvertOdomToTf::ConvertOdomToTf(ros::NodeHandle n) : node(n)
{
    sub = node.subscribe("/odom", 100, &ConvertOdomToTf::odomCallback, this);
}

void ConvertOdomToTf::odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    // tf::TransformBroadcaster odom_tf;
    // tf::Transform transform;
    // transform.setOrigin( tf::Vector3(msg->pose.pose.position.x,msg->pose.pose.position.y, 0.0) );
    // tf::Quaternion q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    // transform.setRotation(q);
    // // odom_tf.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "odom", "base_link"));
    // odom_tf.sendTransform(tf::StampedTransform(transform, msg->header.stamp, "odom", "base_link"));
    // std::cout<<"runnning for publishing odom->base_link tf!  "<<msg->header.stamp<<std::endl;

    static tf2_ros::TransformBroadcaster br;  //发布器
    geometry_msgs::TransformStamped transformStamped;  //发布的数据
    transformStamped.header.stamp = msg->header.stamp;
    transformStamped.header.frame_id = "odom";
    transformStamped.child_frame_id = "base_link";
    transformStamped.transform.translation.x = msg->pose.pose.position.x;
    transformStamped.transform.translation.y = msg->pose.pose.position.y;
    transformStamped.transform.translation.z = msg->pose.pose.position.z;
    transformStamped.transform.rotation = msg->pose.pose.orientation;

    br.sendTransform(transformStamped);  //发布

    // std::cout<<"runnning for publishing odom->base_link tf!  "<<msg->header.stamp<<std::endl;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "convert_tf");
    ros::NodeHandle n;
    ConvertOdomToTf object(n);
    ROS_INFO("convert_tf finished");
    ros::spin();
    return 0;
}