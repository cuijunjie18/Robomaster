#ifndef HKCAM_HPP
#define HKCAM_HPP
// HKSDK
#include <MvCameraControl.h>
// ROS
#include <rm_utils/data.h>

#include <camera_info_manager/camera_info_manager.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <rm_interfaces/msg/rmrobot.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace hik_camera {

struct HikParams {
    std::string sn;
    std::string cam_name;
    std::string robot_name;
    std::string camera_info_url;
    // normal
    double exposure_time;
    double gain;
    double gamma;
    double digital_shift;
    int offset_x;
    int offset_y;
    int out_post_top_offset_y;
    int roi_height;
    int roi_width;
    // energy
    double en_exposure_time;
    double en_gain;
    int en_offset_x;
    int en_offset_y;
    int en_roi_height;
    int en_roi_width;
    std::atomic<bool> en_resize_640;
    double frame_rate;
    bool use_sensor_data_qos;

    // frameinfo
    vision_mode mode;
    std::atomic<Robot_id_dji> robot_id;
    std::atomic<double> bullet_velocity;
    std::atomic<bool> right_press;
    std::atomic<bool> lobshot;
};

class HikCameraNode : public rclcpp::Node {
   public:
    explicit HikCameraNode(const rclcpp::NodeOptions& options);
    ~HikCameraNode() override;

   private:
    int nRet = MV_OK;
    void* camera_handle;
    std::thread capture_thread;
    std::thread monitor_thread;
    HikParams params;
    // 相机图像发布
    image_transport::Publisher image_pub;

    MV_IMAGE_BASIC_INFO img_info;
    MV_CC_PIXEL_CONVERT_PARAM convert_param;
    // 相机发布
    std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager;
    sensor_msgs::msg::CameraInfo camera_info_msg;
    // 接受robot信息
    rclcpp::Subscription<rm_interfaces::msg::Rmrobot>::SharedPtr robot_sub;

    int fail_cnt = 0;
    std::atomic<bool> grab_on = false;
    std::atomic<bool> monitor_on = false;
    std::atomic<bool> camera_failed = false;
    // 只在模式切换的时候更改，避免频繁读写
    std::atomic<vision_mode> grab_vision_mode = AUTO_AIM;
    cv::Mat Gamma_lookUpTable;
    void load_params();
    void init_camera();
    void reset();
    void open_device();
    void close_device();
    void start_grab();
    void stop_grab();
    void set_hk_params();
    void set_grab_params(int offset_x, int offset_y, int roi_width, int roi_height);
    void fit_int_step(std::string property, int& value);
    void grab();
    void monitor();
    void robot_mode_update(rm_interfaces::msg::Rmrobot::ConstSharedPtr msg);
    std::pair<int, int> get_sensor_height_width();
};

class CameraException : public std::exception {
   public:
    std::string info;
    CameraException(const std::string&& _info) : info{_info} {}
    const char* what() const noexcept { return info.c_str(); }
};

}  // namespace hik_camera

#endif