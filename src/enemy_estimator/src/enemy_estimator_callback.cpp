#include <enemy_estimator/enemy_estimator.h>
#include <rm_utils/data.h>
#include <rm_utils/frame_info.h>

#include <rm_utils/perf.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

using namespace enemy_estimator;

void EnemyEstimatorNode::detection_callback(rm_interfaces::msg::Detection::UniquePtr detection_msg) {
    // For test intra comms
    // RCLCPP_INFO(get_logger(), "PID: %d PTR: %p", getpid(),
    // (void*)detection_msg->image.data.data());
    PerfGuard predictor_perf_guard("PredictorPerfTotal");
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

    // 更新坐标变换
    geometry_msgs::msg::TransformStamped t;
    try {
        t = tf2_buffer->lookupTransform(params.target_frame, detection_msg->header.frame_id, detection_msg->header.stamp,
                                        rclcpp::Duration::from_seconds(0.5));
        Eigen::Isometry3d trans_eigen = tf2::transformToEigen(t);
        // std::cout << trans_eigen.matrix() << std::endl;
        pc.update_trans(trans_eigen.matrix());
    } catch (const std::exception& ex) {
        RCLCPP_WARN(this->get_logger(), "Could not transform %s to %s: %s", detection_msg->header.frame_id.c_str(), params.target_frame.c_str(),
                    ex.what());
        return;
    }
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
    states_pub->publish(get_state_msg());

    if (params.enable_imshow) {
        cv::imshow("estimator", result_img);
        if (params.debug && !show_enemies.empty()) {
            cv::imshow("enemy", show_enemies);
        }
        cv::waitKey(1);
    }
}

void EnemyEstimatorNode::robot_callback(rm_interfaces::msg::Rmrobot::SharedPtr robot_msg) {
    // 实时更新imu信息
    imu = robot_msg->imu;
}

rm_interfaces::msg::EnemyStates EnemyEstimatorNode::get_state_msg() {
    rm_interfaces::msg::State state;
    rm_interfaces::msg::EnemyStates msg;
    for (auto& enemy : enemies) {
        state.x = enemy.ekf.Xe[0];
        state.vx = enemy.ekf.Xe[1];
        state.y = enemy.ekf.Xe[2];
        state.vy = enemy.ekf.Xe[3];
        state.theta = enemy.ekf.Xe[4];
        state.w = enemy.ekf.Xe[5];
        state.z = enemy.ekf.Xe[6];
        state.vz = enemy.ekf.Xe[7];
        state.r = enemy.ekf.Xe[8];
        state.last_r = enemy.ekf.last_r;
        state.dz = enemy.dz;
        state.timestamp = enemy.alive_ts;
        msg.states.emplace_back(state);
    }
    return msg;
}