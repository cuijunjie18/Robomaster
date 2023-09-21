#include <rm_republish/rm_repunlish.h>


using namespace rm_republish;

RmRepublishNode::RmRepublishNode(const rclcpp::NodeOptions& options): Node("rm_republish",options){
    RCLCPP_INFO(get_logger(),"Starting Rm Republish Node!");

    use_sensor_data_qos = declare_parameter("use_sensor_data_qos",false);
    camera_name = declare_parameter("camera_name","114514_camera");

    RCLCPP_INFO(get_logger(),"use_sensor_data_qos: %s", use_sensor_data_qos? "True":"False");
    RCLCPP_INFO(get_logger(),"camera_name: %s",camera_name.c_str());

    auto qos = use_sensor_data_qos ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;
    auto imgcb = std::bind(&RmRepublishNode::imgCallback, this, std::placeholders::_1);
    rosbag_sub = image_transport::create_subscription(this, camera_name, imgcb, "compressed",qos);
    img_pub = rclcpp::create_publisher<sensor_msgs::msg::Image>(this,camera_name,rclcpp::SensorDataQoS());
}

void RmRepublishNode::imgCallback(const sensor_msgs::msg::Image::ConstSharedPtr& img_msg){
    static int msg_cnt = 0;
    ++msg_cnt;
    if(msg_cnt % 500 == 0){
        RCLCPP_INFO(get_logger(),"publishing....%d",msg_cnt);
    }
    sensor_msgs::msg::Image::UniquePtr now_msg(new sensor_msgs::msg::Image(*img_msg));
    img_pub->publish(std::move(now_msg));
}


#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_republish::RmRepublishNode)