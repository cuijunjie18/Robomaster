#ifndef _RM_LAGACY_REPLAY_H
#define _RM_LAGACY_REPLAY_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <camera_info_manager/camera_info_manager.hpp>

#include <rm_interfaces/msg/rmrobot.hpp>
#include <rm_utils/data.h>

namespace rm_lagacy_replay {

class ReplayNode: public rclcpp::Node {
public:
    explicit ReplayNode(const rclcpp::NodeOptions& options);
    ~ReplayNode() override;
private:
    // params
    std::string robot_name;
    std::string camera_name;
    std::string video_file;
    std::string imu_file;
    double replay_fps;
    double timestamp_offset;
    std::vector<double> pitch2yaw_t;

    // 相机
    std::string camera_info_url;
    std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager;
    sensor_msgs::msg::CameraInfo camera_info_msg;
    double offset_x, offset_y, en_offset_x, en_offset_y;
    double rx, ry;

    rclcpp::Time start_time;

    // pubs
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
    rclcpp::Publisher<rm_interfaces::msg::Rmrobot>::SharedPtr robot_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub;

    std::thread replay_thread;

    void replay();
    void deal_imu(const recv_msg &msg, double timestamp);
    void deal_img(cv::Mat img, const recv_msg &msg, double timestamp);
};

}; // namespace rm_lagacy_replay

#endif // _RM_LAGACY_REPLAY_H