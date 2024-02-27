#include <enemy_predictor/enemy_predictor.h>
#include <rm_utils/data.h>
#include <rm_utils/frame_info.h>

#include <rm_utils/perf.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

using namespace enemy_predictor;

void EnemyPredictorNode::detection_callback(rm_interfaces::msg::Detection::UniquePtr detection_msg) {
    // For test intra comms
    // RCLCPP_INFO(get_logger(), "PID: %d PTR: %p", getpid(),
    // (void*)detection_msg->image.data.data());
    PerfGuard predictor_perf_guard("PredictorPerfTotal");
    marker_id = 0;
    markers.markers.clear();
    cv::Mat result_img(detection_msg->image.height, detection_msg->image.width, encoding2mat_type(detection_msg->image.encoding),
                       detection_msg->image.data.data());

    FrameInfo frame_info;
    frame_info.deserialize(detection_msg->image.header.frame_id);

    // 更新参数
    params.mode = frame_info.mode;
    params.right_press = frame_info.right_press;
    params.lobshot = frame_info.lobshot;
    // 更新相机参数
    pc.update_camera_info(frame_info.k, frame_info.d);

    // 画一下画面中心(光心)
    if (params.debug || params.enable_imshow) {
        cv::circle(result_img, cv::Point(frame_info.k[2], frame_info.k[5]), 3, cv::Scalar(255, 0, 255), 2);
    }
    // 更新Position_Calculator
    pc.update_tf(tf2_buffer, detection_msg->header);

    int now_id = get_rmcv_id(frame_info.robot_id);
    RCLCPP_INFO(get_logger(), "robot_id: %d", now_id);
    if (now_id != params.rmcv_id) {
        params.rmcv_id = now_id;
        RCLCPP_INFO(get_logger(), "switch to %d", params.rmcv_id);
    }
    if (now_id == UNKNOWN_ID) {
        RCLCPP_WARN(get_logger(), "unknown id %d", now_id);
        return;
    }
    if (frame_info.bullet_velocity > 8.) {  // 只相信大于8的数据
        bac->refresh_velocity(params.rmcv_id % 9 == 1, frame_info.bullet_velocity);
    }

    recv_detection.img = result_img;
    recv_detection.mode = params.mode;
    recv_detection.time_stamp = rclcpp::Time(detection_msg->image.header.stamp).seconds();
    recv_detection.res.clear();
    for (int i = 0; i < (int)detection_msg->detected_armors.size(); ++i) {
        recv_detection.res.emplace_back(Msg2Armor(detection_msg->detected_armors[i]));
    }

    update_armors();
    update_enemy();

    ControlMsg now_cmd = get_command();
    // if (now_cmd.flag != 0 && params.disable_auto_shoot) {
    //     now_cmd.flag = 1;
    // }
    // now_cmd.header.frame_id = "robot_cmd: " + std::to_string(frame_info.robot_id);
    // now_cmd.header.stamp = detection_msg->header.stamp;
    // control_pub->publish(now_cmd);

    show_enemies_pub->publish(markers);
    if (params.enable_imshow) {
        cv::imshow("predictor", result_img);
        // if (params.debug && !show_enemies.empty()) {
        //     cv::imshow("enemy", show_enemies);
        // }
        cv::waitKey(1);
    }
}

void EnemyPredictorNode::robot_callback(rm_interfaces::msg::Rmrobot::SharedPtr robot_msg) {
    // 实时更新imu信息
    imu = robot_msg->imu;
}