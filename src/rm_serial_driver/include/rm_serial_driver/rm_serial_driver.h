#ifndef _RM_SERIAL_DRIVER_H
#define _RM_SERIAL_DRIVER_H

#include <rm_utils/data.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include <rm_interfaces/msg/rmrobot.hpp>
#include <serial/serial.h>
#include <std_msgs/msg/float64.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker.hpp>

// C++ system
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace rm_serial_driver {
class RMSerialDriver : public rclcpp::Node {
   public:
    explicit RMSerialDriver(const rclcpp::NodeOptions& options);

    ~RMSerialDriver() override;
    static constexpr int robotpub_count_down = 100;
   private:
    // Serial port
    std::unique_ptr<serial::Serial> serial_driver;

    // Broadcast tf from odom to gimbal_link
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

    // robot mode & id publish
    rclcpp::Publisher<rm_interfaces::msg::Rmrobot>::SharedPtr robot_pub;
    // control sub
    rclcpp::Subscription<rm_interfaces::msg::Control>::SharedPtr control_sub;

    std::thread receive_thread;

    // params
    float timestamp_offset;
    int baud_rate;
    int imu_msg_id;
    std::string device_name;
    std::string robot_name;
    std::vector<double> pitch2yaw_t;

    void getParams();
    void receiveData();
    void reopenPort();
    void imuMsgCallback(recv_msg* msg);
    void robotpub_low_freq(const rm_interfaces::msg::Rmrobot& msg);
    void ControlMsgCallback(rm_interfaces::msg::Control::ConstSharedPtr control_msg);
};
};  // namespace rm_serial_driver

#endif