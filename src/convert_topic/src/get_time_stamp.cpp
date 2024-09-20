#include "convert_topic/get_time_stamp.h"

GetTimeStamp::GetTimeStamp(ros::NodeHandle n) : node(n)
{
    sub = node.subscribe("/scan", 1, &GetTimeStamp::scanCallback, this);
}

void GetTimeStamp::scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
{
    std::string out_filename = "/home/shiwb/timestamp.txt";
    std::ofstream f_out;
    f_out.open(out_filename.c_str(), std::ios::app);
    f_out << std::fixed;	
    f_out << std::setprecision(9) << msg->header.stamp.toSec() << std::endl;
    f_out.close();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "get_time_stamp");
    ros::NodeHandle n;
    GetTimeStamp object(n);
    ros::spin();
    return 0;
}
