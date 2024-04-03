#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"

rclcpp::Rate rate(50);

namespace recorder {
class RecorderNode : public rclcpp::Node {
   public:
    // 使用RCLCPP_COMPONENTS_REGISTER_NODE时，需要一个显式的构造函数
    explicit RecorderNode(const rclcpp::NodeOptions &options) : Node("recorder_node", options) {
        RCLCPP_INFO(get_logger(), "START RECORD");
        std::string image_topic_name =
            this->declare_parameter<std::string>("image_topic_name", "/image_topic");
        std::string video_save_path =
            this->declare_parameter<std::string>("video_save_path", "./");  // 默认保存在当前目录

        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y_%m_%d_%H_%M_%S");
        std::string file_name = "record_" + ss.str() + ".avi";

        video_save_path = video_save_path + "/" + file_name;  // 使用 /= 操作符添加文件名
        video_writer = cv::VideoWriter(video_save_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                       50, cv::Size(640, 640));

        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            image_topic_name, rclcpp::SensorDataQoS(),
            std::bind(&RecorderNode::image_callback, this, std::placeholders::_1));
    }

   private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            video_writer.write(cv_ptr->image);
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        rate.sleep();
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    cv::VideoWriter video_writer;
};

}  // namespace recorder

// 使用RCLCPP_COMPONENTS_REGISTER_NODE宏来注册节点作为组件
RCLCPP_COMPONENTS_REGISTER_NODE(recorder::RecorderNode)
