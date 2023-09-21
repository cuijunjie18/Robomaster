#include <cv_bridge/cv_bridge.h>
#include <detector/detector_node.h>
#include <rm_utils/frame_info.h>
#include <rm_utils/common.h>

#include <rm_utils/perf.hpp>
using namespace rm_detector;

DetectorNode::DetectorNode(const rclcpp::NodeOptions& options) : Node("rm_detector", options) {
    RCLCPP_INFO(this->get_logger(), "Starting Robomaster Detector Node!");

    bool use_intra = options.use_intra_process_comms();
    if (!use_intra) {
        RCLCPP_WARN(get_logger(), "Not In Intra Process Mode");
    }

    // load params
    params.camera_name = declare_parameter("camera_name", "cam_raw");
    params.robot_name = declare_parameter("robot_name", "15robot");
    params.node_dir = declare_parameter("detector_dir", "");
    params.armor_config = declare_parameter("armor_detector_config", "");
    params.energy_config = declare_parameter("energy_detector_config", "");
    params.enable_imshow = declare_parameter("enable_imshow", false);
    params.debug = declare_parameter("debug", false);

    RCLCPP_INFO(this->get_logger(), "camera_name: %s", params.camera_name.c_str());
    RCLCPP_INFO(this->get_logger(), "robot_name: %s", params.robot_name.c_str());
    RCLCPP_INFO(this->get_logger(), "share_dir: %s", params.node_dir.c_str());
    RCLCPP_INFO(this->get_logger(), "armor_config: %s", params.armor_config.c_str());
    RCLCPP_INFO(this->get_logger(), "energy_config: %s", params.energy_config.c_str());
    RCLCPP_INFO(this->get_logger(), "enable_imshow: %s", params.enable_imshow ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "debug: %s", params.debug ? "true" : "false");

    bool use_trad = false;
    auto armor_toml_config = toml::parse(params.node_dir + "/" + params.armor_config);
    auto type = armor_toml_config.at("type").as_string();
    if (type == "Trad") {
        use_trad = true;
        detector_armor = std::make_shared<DetectorTrad>(params.armor_config, params.node_dir,
                                                        this->get_logger());
    }

    // build detectors
#ifdef TRT
    cuInit(0);

    CUresult res = cuDeviceGet(&cuda_device, 0);
    RCLCPP_WARN(get_logger(), "CUDA DEVICE: %p res: %d", (void*)&cuda_device, res);
    res = cuCtxCreate(&thread_ctx, 0, cuda_device);
    RCLCPP_WARN(get_logger(), "CUDA CREATE CTX %p res: %d", (void*)&thread_ctx, res);

    if (!use_trad) {
        detector_armor =
            std::make_shared<DetectorTRT>(params.armor_config, params.node_dir, this->get_logger());
    }
    detector_energy =
        std::make_shared<DetectorTRT>(params.energy_config, params.node_dir, this->get_logger());

    if (!use_trad) {
        std::dynamic_pointer_cast<DetectorTRT>(detector_armor)->set_cuda_context(&thread_ctx);
    }
    std::dynamic_pointer_cast<DetectorTRT>(detector_energy)->set_cuda_context(&thread_ctx);

#else
    if (!use_trad) {
        detector_armor = std::make_shared<DetectorVINO>(params.armor_config, params.node_dir,
                                                        this->get_logger());
    }
    detector_energy =
        std::make_shared<DetectorVINO>(params.energy_config, params.node_dir, this->get_logger());
#endif

    // 注册sub/pub
    img_sub = this->create_subscription<sensor_msgs::msg::Image>(
        params.camera_name, rclcpp::SensorDataQoS(),
        std::bind(&DetectorNode::image_callback, this, std::placeholders::_1));
    detection_armor_pub = this->create_publisher<rm_interfaces::msg::Detection>(
        "/detection_armor", rclcpp::SensorDataQoS());
    detection_energy_pub = this->create_publisher<rm_interfaces::msg::Detection>(
        "/detection_energy", rclcpp::SensorDataQoS());
    // Debug pub
    binary_img_pub = image_transport::create_publisher(this, "/rm_detector/binary");
    number_img_pub = image_transport::create_publisher(this, "/rm_detector/number_img");
    result_img_pub = image_transport::create_publisher(this, "/rm_detector/result_img");
    lights_data_pub =
        this->create_publisher<rm_interfaces::msg::DebugLights>("/rm_detector/debug_lights", 10);
    armors_data_pub =
        this->create_publisher<rm_interfaces::msg::DebugArmors>("/rm_detector/debug_armors", 10);
}

void DetectorNode::image_callback(sensor_msgs::msg::Image::UniquePtr img_msg) {
    // auto img = cv_bridge::toCvShare(img_msg, "bgr8")->image;
    PerfGuard detector_perf_guard("DetectorTotal");
    cv::Mat img(img_msg->height, img_msg->width, encoding2mat_type(img_msg->encoding),
                img_msg->data.data());
    FrameInfo frame_info;
    try {
        frame_info.deserialize(img_msg->header.frame_id);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Fail to deserialize frame info: %s", e.what());
        rclcpp::shutdown();
        return;
    }
    params.mode = frame_info.mode;
    // For test intra comms
    // RCLCPP_INFO(get_logger(),"PID: %d PTR: %p",getpid(),(void*)img.data);

    if (!detector_armor || !detector_energy) {
        RCLCPP_ERROR(this->get_logger(), "Invalid Detector!");
        rclcpp::shutdown();
    }

    std::vector<Armor> res;
    std::shared_ptr<Detector> now_detector;
    if (params.mode == B_WM || params.mode == S_WM) {
        res = detector_energy->detect(img);
        now_detector = detector_energy;
    } else {
        res = detector_armor->detect(img);
        now_detector = detector_armor;
    }
    // RCLCPP_INFO(get_logger(),"res size: %d",res.size());

    if (params.debug || params.enable_imshow) {
        now_detector->draw(img, res);
    }

    if (params.debug) {
        // 发布调试信息
        std::shared_ptr<DetectorTrad> now_trad =
            std::dynamic_pointer_cast<DetectorTrad>(now_detector);
        if (now_trad) {
            binary_img_pub.publish(
                cv_bridge::CvImage(img_msg->header, "mono8", now_trad->binary_img).toImageMsg());
            auto all_num_img = now_trad->getAllNumbersImage();
            number_img_pub.publish(
                cv_bridge::CvImage(img_msg->header, "mono8", all_num_img).toImageMsg());

            // Sort lights and armors data by x coordinate
            std::sort(now_trad->debug_lights.data.begin(), now_trad->debug_lights.data.end(),
                      [](const auto& l1, const auto& l2) { return l1.center_x < l2.center_x; });
            std::sort(now_trad->debug_armors.data.begin(), now_trad->debug_armors.data.end(),
                      [](const auto& a1, const auto& a2) { return a1.center_x < a2.center_x; });

            lights_data_pub->publish(now_trad->debug_lights);
            armors_data_pub->publish(now_trad->debug_armors);
        }
        result_img_pub.publish(cv_bridge::CvImage(img_msg->header, "bgr8", img).toImageMsg());
    }

    if (params.enable_imshow) {
        cv::imshow("detect", img);
        cv::waitKey(1);
    }
    // 二次包装
    rm_interfaces::msg::Detection::UniquePtr detection_msg(new rm_interfaces::msg::Detection());
    // std::move 实现data资源零拷贝
    detection_msg->header = img_msg->header;
    // 更改frame_id 为了之后tf2_filter
    detection_msg->header.frame_id = "camera_optical_frame";
    detection_msg->image.header = img_msg->header;
    detection_msg->image.encoding = img_msg->encoding;
    detection_msg->image.height = img_msg->height;
    detection_msg->image.width = img_msg->width;
    detection_msg->image.data = std::move(img_msg->data);
    img_msg.reset();
    for (size_t i = 0; i < res.size(); ++i) {
        detection_msg->detected_armors.emplace_back(Armor2Msg(res[i]));
    }
    if (params.mode == B_WM || params.mode == S_WM) {
        detection_energy_pub->publish(std::move(detection_msg));
    } else {
        detection_armor_pub->publish(std::move(detection_msg));
    }
}

DetectorNode::~DetectorNode() {}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_detector::DetectorNode)
