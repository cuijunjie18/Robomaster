#ifndef RMCV_POSCALC_HPP_
#define RMCV_POSCALC_HPP_

#include <tf2_ros/buffer.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <vector>

// x轴朝前、y轴朝左、z轴朝上
// roll 从y轴转向z轴为正 pitch 向下为正，yaw向左转为正

class Position_Calculator {
   private:
    Eigen::Matrix<double, 3, 3> K;  // 内参矩阵
    Eigen::Matrix<double, 1, 5> D;  // 畸变矩阵
    cv::Mat Kmat, Dmat;
    std::shared_ptr<tf2_ros::Buffer> tf2_buffer;
    std_msgs::msg::Header detection_header;

   public:
    // pnp解算结果
    struct pnp_result {
        Eigen::Matrix<double, 3, 1> xyz;
        // Eigen::Matrix<double, 3, 1> Rvec;
        Eigen::Vector3d normal_vec;  // 法向量
        Eigen::Vector3d show_vec;    // 显示的法向量
    };
    static std::vector<cv::Vec3d> SmallArmor, BigArmor, pw_energy, pw_result;

    void update_camera_info(const std::vector<double>& k_, const std::vector<double>& d_);
    void update_tf(std::shared_ptr<tf2_ros::Buffer> tf2_buffer_,
                   std_msgs::msg::Header detection_header_);
    Eigen::Vector3d trans(const std::string& target_frame, const std::string& source_frame,
                          Eigen::Vector3d source_point);
    pnp_result pnp(const std::vector<cv::Point2d> pts, bool isBigArmor);
    cv::Point2d pos2img(Eigen::Matrix<double, 3, 1> X);

    // void update_trans(const Eigen::Matrix<double, 4, 4>& trans_);
    // Eigen::Matrix<double, 3, 1> pb_to_pc(Eigen::Matrix<double, 3, 1> pb);
    // Eigen::Matrix<double, 3, 1> pc_to_pb(Eigen::Matrix<double, 3, 1> pc);
    // Eigen::Matrix<double, 3, 1> pnp_get_pb(const std::vector<cv::Point2d> pts, bool isBigArmor);
    // std::vector<Eigen::Vector3d> pnp_get_pb_WM(const std::vector<cv::Point2d> pts);  // （对外）
};

#endif