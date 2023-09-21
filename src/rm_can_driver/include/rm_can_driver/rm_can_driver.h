#ifndef _RMCV_CAN_DRIVER_H
#define _RMCV_CAN_DRIVER_H

#include <rm_utils/data.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include <rm_interfaces/msg/rmrobot.hpp>
#include <std_msgs/msg/float64.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <rm_can_driver/socket_can_receiver.hpp>
#include <rm_can_driver/socket_can_sender.hpp>
#include <rm_can_driver/msg_receiver.h>
namespace rm_can_driver {
class RMCanDriver : public rclcpp::Node {
   public:
    explicit RMCanDriver(const rclcpp::NodeOptions& options);

   private:
    // CanDevice
    std::unique_ptr<drivers::socketcan::SocketCanReceiver> can_receiver;
    std::unique_ptr<drivers::socketcan::SocketCanSender> can_sender;
    drivers::socketcan::SocketCanReceiver::CanFilterList filter_list;
    // data receiver
    std::unique_ptr<msg_receiver<recv_msg>> rm_receiver;

    // Broadcast tf from odom to gimbal_link
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

    // robot mode & id publish
    rclcpp::Publisher<rm_interfaces::msg::Rmrobot>::SharedPtr robot_pub;
    
    // robot control sub
    rclcpp::Subscription<rm_interfaces::msg::Control>::SharedPtr control_sub;

    std::thread receive_thread;

    // params
    float timestamp_offset;
    int recv_id;
    int send_id;
    int baud_rate;
    std::string device_name;
    std::string robot_name;
    std::string sudo_pwd;
    std::vector<double> pitch2yaw_t;

    void loadParams();
    void receiveData();
    void reopenPort();
    void imuMsgCallback(recv_msg* msg);
    void ControlMsgCallback(rm_interfaces::msg::Control::ConstSharedPtr control_msg);
};
};  // namespace rm_can_driver

#endif