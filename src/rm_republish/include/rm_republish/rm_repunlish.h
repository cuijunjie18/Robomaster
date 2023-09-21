#ifndef _RM_REPUBLISH_H
#define _RM_REPUBLISH_H

#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace rm_republish {
class RmRepublishNode : public rclcpp::Node {
   public:
    explicit RmRepublishNode(const rclcpp::NodeOptions& options);

   private:
    bool use_sensor_data_qos;
    std::string camera_name;
    // subscribe

    image_transport::Subscriber rosbag_sub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub;

    void imgCallback(const sensor_msgs::msg::Image::ConstSharedPtr& img_msg); 
};
};  // namespace rm_republish

#endif