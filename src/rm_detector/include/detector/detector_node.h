#ifndef _DETECTOR_NODE_H
#define _DETECTOR_NODE_H
// ROS
#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <rm_interfaces/msg/detection.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

// detector
#include <detector/detector.h>
#include <detector/detector_trad.h>
#include <detector/net_decoder.h>

#ifdef TRT
#include <detector/detector_trt.h>
#else
#include <detector/detector_vino.h>
#endif

namespace rm_detector {

struct DetectorNodeParams {
    std::string camera_name;
    std::string robot_name;
    std::string node_dir;
    std::string armor_config;
    std::string energy_config;
    vision_mode mode;
    bool enable_imshow;
    bool debug;
};

class DetectorNode : public rclcpp::Node {
   private:
    DetectorNodeParams params;
    std::shared_ptr<Detector> detector_armor;
    std::shared_ptr<Detector> detector_energy;

#ifdef TRT
    CUcontext thread_ctx;
    CUdevice cuda_device;
#endif

    // Image Sub
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub;

    // Detected armors publisher
    rclcpp::Publisher<rm_interfaces::msg::Detection>::SharedPtr detection_armor_pub;
    rclcpp::Publisher<rm_interfaces::msg::Detection>::SharedPtr detection_energy_pub;

    // Debug Image Pub
    image_transport::Publisher binary_img_pub;
    image_transport::Publisher number_img_pub;
    image_transport::Publisher result_img_pub;

    // Debug Information Pub
    rclcpp::Publisher<rm_interfaces::msg::DebugLights>::SharedPtr lights_data_pub;
    rclcpp::Publisher<rm_interfaces::msg::DebugArmors>::SharedPtr armors_data_pub;

    void image_callback(sensor_msgs::msg::Image::UniquePtr img_msg);

   public:
    explicit DetectorNode(const rclcpp::NodeOptions& options);
    ~DetectorNode() override;
};
};  // namespace rm_detector

#endif