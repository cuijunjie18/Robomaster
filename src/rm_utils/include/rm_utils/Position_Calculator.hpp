#ifndef RMCV_POSCALC_HPP_
#define RMCV_POSCALC_HPP_

#include <tf2_ros/buffer.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <rm_utils/perf.hpp>
#include <std_msgs/msg/float64.hpp>
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
        double yaw;
        std::vector<cv::Point2d> img_pts;
    };
    static std::vector<cv::Vec3d> SmallArmor, BigArmor, pw_energy, pw_result;

    void update_camera_info(const std::vector<double>& k_, const std::vector<double>& d_);
    void update_tf(std::shared_ptr<tf2_ros::Buffer> tf2_buffer_,
                   std_msgs::msg::Header detection_header_);
    Eigen::Vector3d trans(const std::string& target_frame, const std::string& source_frame,
                          Eigen::Vector3d source_point);

    // 根据给定的pitch和yaw生成roll为零的装甲板系某点在odom系的坐标，pitch和yaw为角度制
    Eigen::Vector3d generate_armor_point_odom(double pitch, double yaw, Eigen::Vector3d xyz,
                                              Eigen::Vector3d point_armor);
    // 根据给定的pitch和yaw生成roll为零的装甲板在图像上的投影，包括角点和中心点，pitch和yaw为角度制
    std::vector<cv::Point2d> generate_armor_img(bool isBigArmor, double pitch, double yaw,
                                                Eigen::Vector3d xyz);
    double diff_fun_nor_dis(std::vector<cv::Point2d> img_pts, Eigen::Vector3d xyz,
                            std::vector<cv::Point2d> guess_pts);
    double diff_fun_side_angle(std::vector<cv::Point2d> img_pts, Eigen::Vector3d xyz,
                               std::vector<cv::Point2d> guess_pts);
    double diff_fun_area(std::vector<cv::Point2d> img_pts, Eigen::Vector3d xyz,
                         std::vector<cv::Point2d> guess_pts);
    double diff_fun_left_right_ratio(std::vector<cv::Point2d> img_pts, Eigen::Vector3d xyz,
                                     std::vector<cv::Point2d> guess_pts);
    double final_diff_fun_cal(bool isBigArmor, std::vector<cv::Point2d> img_pts,
                              Eigen::Vector3d xyz, double pitch, double yaw);
    double final_diff_fun_choose(bool isBigArmor, std::vector<cv::Point2d> img_pts,
                                 Eigen::Vector3d xyz, double pitch, double yaw);
    pnp_result pnp(const std::vector<cv::Point2d> pts, bool isBigArmor);
    // pnp_result rm_pnp(const std::vector<cv::Point2d> pts, bool isBigArmor,
    //                   std::vector<rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr>
    //                   watch_pub);
    pnp_result rm_pnp(const std::vector<cv::Point2d> pts, bool isBigArmor);
    cv::Point2d pos2img(Eigen::Matrix<double, 3, 1> X);

    // void update_trans(const Eigen::Matrix<double, 4, 4>& trans_);
    // Eigen::Matrix<double, 3, 1> pb_to_pc(Eigen::Matrix<double, 3, 1> pb);
    // Eigen::Matrix<double, 3, 1> pc_to_pb(Eigen::Matrix<double, 3, 1> pc);
    // Eigen::Matrix<double, 3, 1> pnp_get_pb(const std::vector<cv::Point2d> pts, bool isBigArmor);
    // std::vector<Eigen::Vector3d> pnp_get_pb_WM(const std::vector<cv::Point2d> pts);  // （对外）
};

#endif