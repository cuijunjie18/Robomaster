#include <rm_lagacy_replay/rm_lagacy_replay.h>

#include <fstream>
#include <opencv2/opencv.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cv_bridge/cv_bridge.h>

#include <rm_utils/frame_info.h>

using namespace rm_lagacy_replay;

ReplayNode::ReplayNode(const rclcpp::NodeOptions& options)
    :Node("rm_lagacy_replay", options)
{
    RCLCPP_INFO(get_logger(),"Starting Rm Lagacy Replay Node!");

    robot_name = declare_parameter("robot_name", "virtual_robot");
    camera_name = declare_parameter("camera_name", "virtual_camera");
    video_file = declare_parameter("video_file", "test.mp4");
    imu_file = declare_parameter("imu_file", "test.imu");
    replay_fps = declare_parameter("replay_fps", 60.0);

    timestamp_offset = declare_parameter("timestamp_offset", 0.0);
    pitch2yaw_t = declare_parameter("pitch2yaw_t", std::vector<double>{0.0, 0.0, 0.0});

    start_time = this->now();

    RCLCPP_INFO(get_logger(), "robot_name: %s", robot_name.c_str());
    RCLCPP_INFO(get_logger(), "video_file: %s", video_file.c_str());
    RCLCPP_INFO(get_logger(), "imu_file: %s", imu_file.c_str());

    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    robot_pub = create_publisher<rm_interfaces::msg::Rmrobot>(robot_name, rclcpp::SensorDataQoS());
    img_pub = create_publisher<sensor_msgs::msg::Image>(camera_name, rclcpp::SensorDataQoS());

    // load camera info
    camera_info_url = declare_parameter("camera_info_url", std::string(""));
    camera_info_manager =
        std::make_unique<camera_info_manager::CameraInfoManager>(this, camera_name);
    if (camera_info_manager->validateURL(camera_info_url)) {
        camera_info_manager->loadCameraInfo(camera_info_url);
        camera_info_msg = camera_info_manager->getCameraInfo();
    } else {
        RCLCPP_WARN(this->get_logger(), "Invalid camera info URL: %s",
                    camera_info_url.c_str());
    }
    offset_x = declare_parameter("offset_x", 0.0);
    offset_y = declare_parameter("offset_y", 0.0);
    en_offset_x = declare_parameter("en_offset_x", 0.0);
    en_offset_y = declare_parameter("en_offset_y", 0.0);

    replay_thread = std::thread(&ReplayNode::replay, this);
}

ReplayNode::~ReplayNode()
{
    if(replay_thread.joinable())
    {
        replay_thread.join();
    }
}

void ReplayNode::deal_imu(const recv_msg &msg, double timestamp)
{
    tf2::Quaternion pitch2yaw_r, yaw2odom_r;
    pitch2yaw_r.setRPY(msg.roll, msg.pitch, 0);
    yaw2odom_r.setRPY(0, 0, msg.yaw);
    tf2::Transform pitch2yaw(pitch2yaw_r,
                             tf2::Vector3(pitch2yaw_t[0], pitch2yaw_t[1], pitch2yaw_t[2]));
    tf2::Transform yaw2odom(yaw2odom_r);
    tf2::Transform gimbal2odom = yaw2odom * pitch2yaw;

    geometry_msgs::msg::TransformStamped t;
    timestamp_offset = this->get_parameter("timestamp_offset").as_double();
    t.header.stamp = start_time + rclcpp::Duration::from_seconds(timestamp + timestamp_offset);
    t.header.frame_id = "odom";
    t.child_frame_id = "gimbal_link";
    t.transform = tf2::toMsg(gimbal2odom);
    tf_broadcaster->sendTransform(t);

    rm_interfaces::msg::Rmrobot robot_msg;
    robot_msg.robot_id = msg.robot_id;
    robot_msg.bullet_velocity = msg.bullet_speed;
    robot_msg.vision_mode = mode2string(cast_run_mode(msg.mode));
    robot_msg.right_press = msg.mode == 1;
    robot_msg.lobshot = msg.mode == 4;
    robot_msg.imu.roll = msg.roll;
    robot_msg.imu.pitch = msg.pitch;
    robot_msg.imu.yaw = msg.yaw;
    // robotpub_low_freq(robot_msg);
    // robot_msg.robot_id = 5;
    robot_pub->publish(robot_msg);
}

void ReplayNode::deal_img(cv::Mat img, const recv_msg &msg, double timestamp)
{
    cv_bridge::CvImage img_msg (
        std_msgs::msg::Header(),
        sensor_msgs::image_encodings::BGR8,
        img
    );
    
    FrameInfo cam_frame_info;
    // load cam_frame_info
    vision_mode mode = cast_run_mode(msg.mode);
    cam_frame_info.mode = mode;
    cam_frame_info.d = camera_info_msg.d;
    cam_frame_info.k = std::vector<double>(camera_info_msg.k.begin(),camera_info_msg.k.end());
    cam_frame_info.robot_id = static_cast<Robot_id_dji>(msg.robot_id);
    cam_frame_info.bullet_velocity = msg.bullet_speed;
    cam_frame_info.right_press = false;
    cam_frame_info.lobshot = false;

    if(msg.mode == B_WM || msg.mode == S_WM) {
        cam_frame_info.k[2] -= en_offset_x;
        cam_frame_info.k[5] -= en_offset_y;
    } else {
        cam_frame_info.k[2] -= offset_x;
        cam_frame_info.k[5] -= offset_y;
    }
    cam_frame_info.k[0] *= rx;
    cam_frame_info.k[4] *= ry;
    cam_frame_info.k[2] *= rx;
    cam_frame_info.k[5] *= ry;

    img_msg.header.stamp = start_time + rclcpp::Duration::from_seconds(timestamp);
    img_msg.header.frame_id = cam_frame_info.serialize();
    
    img_pub->publish(*img_msg.toImageMsg());
}

void ReplayNode::replay()
{
    cv::VideoCapture cap(video_file);
    std::ifstream imu_in(imu_file);
    if(!cap.isOpened()) {
        RCLCPP_ERROR(get_logger(), "Error opening video file: %s", video_file.c_str());
        rclcpp::shutdown();
    }
    if(!imu_in.is_open()) {
        RCLCPP_ERROR(get_logger(), "Error opening imu file: %s", imu_file.c_str());
        rclcpp::shutdown();
    }
    double local_fps = replay_fps;
    rclcpp::Rate::SharedPtr loop_rate = std::make_shared<rclcpp::Rate>(local_fps);
    recv_msg imu;
    for(; rclcpp::ok(); loop_rate->sleep()) {
        local_fps = this->get_parameter("replay_fps").as_double();
        if(local_fps <= 0.0) { // pause
            RCLCPP_INFO(get_logger(), "Pause");
            continue;
        }
        if(local_fps != replay_fps) {
            replay_fps = local_fps;
            loop_rate = std::make_shared<rclcpp::Rate>(local_fps);
        }

        cv::Mat frame;
        cap >> frame;
        
        if(frame.empty()) {
            RCLCPP_WARN(get_logger(), "Video file end, replay");
            cap.release();
            imu_in.close();
            cap.open(video_file);
            imu_in.open(imu_file);
            cap >> frame;
            if(!cap.isOpened() || !imu_in.is_open() || frame.empty()) {
                RCLCPP_ERROR(get_logger(), "Error opening video file: %s", video_file.c_str());
                rclcpp::shutdown();
            }

            start_time = this->now();
        }
        
        double timestamp;
        int tmp_id;
        imu_in >> imu.roll >> imu.pitch >> imu.yaw >> imu.mode >> tmp_id >> imu.bullet_speed >> timestamp;
        imu.robot_id = tmp_id;
        deal_imu(imu, timestamp);

        deal_img(frame, imu, timestamp);

        RCLCPP_INFO(get_logger(), "robot_id: %d, timestamp: %lf", imu.robot_id, timestamp);
    }
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_lagacy_replay::ReplayNode)