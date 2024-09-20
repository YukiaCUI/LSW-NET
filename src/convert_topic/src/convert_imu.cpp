#include "convert_topic/convert_imu.h"

ConvertImu::ConvertImu(ros::NodeHandle n) : node(n)
{
    sub = node.subscribe("/Yesense/imu_data2", 1, &ConvertImu::imuCallback, this);
    pub = node.advertise<sensor_msgs::Imu>("/Yesense/imu_data", 1);
}

void ConvertImu::imuCallback(const sensor_msgs::Imu::ConstPtr& msg)
{
    sensor_msgs::Imu imu;
    imu.header.stamp = ros::Time::now();
    imu.header.frame_id = "imu";
    imu.orientation = msg->orientation;
    imu.orientation_covariance = msg->orientation_covariance;
    imu.angular_velocity = msg->angular_velocity;
    imu.angular_velocity_covariance = msg->angular_velocity_covariance;
    imu.linear_acceleration = msg->linear_acceleration;
    imu.linear_acceleration_covariance = msg->linear_acceleration_covariance;

    pub.publish(imu);
    ROS_INFO("publish complete");
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "convert_imu");
    ros::NodeHandle n;
    ConvertImu imuObject(n);
    ros::spin();
    return 0;
}