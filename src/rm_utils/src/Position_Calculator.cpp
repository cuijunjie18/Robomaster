#include <rm_utils/Position_Calculator.hpp>

std::vector<cv::Vec3d> Position_Calculator::SmallArmor = {
    // 单位 米
    {-0.0675, 0.0275, 0.},
    {-0.0675, -0.0275, 0.},
    {0.0675, -0.0275, 0.},
    {0.0675, 0.0275, 0.},
};
std::vector<cv::Vec3d> Position_Calculator::BigArmor = {
    {-0.115, 0.029, 0.},
    {-0.115, -0.029, 0.},
    {0.115, -0.029, 0.},
    {0.115, 0.029, 0.},
};
std::vector<cv::Vec3d> Position_Calculator::pw_energy = {  // 单位：m
    {-0.1542, -0.15456, 0.},
    {0.1542, -0.15456, 0.},
    {0.18495, 0.15839, 0.},
    {0., 0.52879, 0.},
    {-0.18495, 0.15839, 0.}};
std::vector<cv::Vec3d> Position_Calculator::pw_result = {  // 单位：m
    {-0.18495, 0.15839, 0.},
    {-0.1542, -0.15456, 0.},
    {0.1542, -0.15456, 0.},
    {0.18495, 0.15839, 0.},
    {0., 0.7, 0.}};

void Position_Calculator::update_tf(std::shared_ptr<tf2_ros::Buffer> tf2_buffer_,
                                    std_msgs::msg::Header detection_header_) {
    tf2_buffer = tf2_buffer_;
    detection_header = detection_header_;
}

void Position_Calculator::update_camera_info(const std::vector<double> &k_,
                                             const std::vector<double> &d_) {
    assert(k_.size() == 9 && "Camera Matrix must has 9 elements.");
    assert(d_.size() == 5 && "Distortion Matrix must has 9 elements.");
    K = Eigen::Matrix<double, 3, 3>(k_.data()).transpose();
    D = Eigen::Matrix<double, 1, 5>(d_.data());
    cv::eigen2cv(K, Kmat);
    cv::eigen2cv(D, Dmat);
}

/** \brief 给定起点和终点的frame_id，计算坐标变换
 * \param target_frame 终点的frame_id
 * \param source_frame 起点的frame_id
 * \param source_point 待转换的坐标
 * \return 转换后的坐标
 */
Eigen::Vector3d Position_Calculator::trans(const std::string &target_frame,
                                           const std::string &source_frame,
                                           Eigen::Vector3d source_point) {
    geometry_msgs::msg::TransformStamped t;
    try {
        t = tf2_buffer->lookupTransform(target_frame, source_frame, detection_header.stamp,
                                        rclcpp::Duration::from_seconds(0.5));
    } catch (const std::exception &ex) {
        printf("Could not transform %s to %s: %s", source_frame, target_frame.c_str(), ex.what());
        abort();
    }
    Eigen::Isometry3d trans_eigen = tf2::transformToEigen(t);
    Eigen::Matrix<double, 4, 1> result;
    result << source_point[0], source_point[1], source_point[2], 1.;
    result = trans_eigen * result;
    return result.block<3, 1>(0, 0);
}

std::vector<cv::Point2d> Position_Calculator::generate_armor_img(bool isBigArmor, double pitch,
                                                                 double yaw, Eigen::Vector3d xyz) {
    std::vector<cv::Point2d> armor_img;
    std::vector<cv::Vec3d> armor_pts;
    double pitch_rad = pitch / 180 * M_PI;
    double yaw_rad = yaw / 180 * M_PI;
    if (isBigArmor) {
        armor_pts = BigArmor;
    } else {
        armor_pts = SmallArmor;
    }
    for (int i = 0; i < armor_pts.size(); ++i) {
        // armor系转odom系
        Eigen::Vector3d odom_point;
        cv::cv2eigen(armor_pts[i], odom_point);
        // 坐标转换统一表示为基变换
        Eigen::Matrix3d
            target2armor_R;  // target系是坐标轴方向定义方式与odom系统一的装甲板系，便于处理pitch和yaw
        target2armor_R << 0, 0, 1,  //
            1, 0, 0,                //
            0, 1, 0;                //
        odom_point = target2armor_R * odom_point;
        Eigen::Matrix3d odom2target_R;
        odom2target_R = Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitZ()) *
                        Eigen::AngleAxisd(pitch_rad, Eigen::Vector3d::UnitY()) *
                        Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX());
        odom_point = odom2target_R * odom_point + xyz;

        // odom_point = trans("yaw_link", "odom", odom_point);
        std::cout << "odom_point: " << odom_point << std::endl;
        armor_img.push_back(pos2img(odom_point));
    }
    armor_img.push_back(pos2img(xyz));
    return armor_img;
}

Position_Calculator::pnp_result Position_Calculator::pnp(const std::vector<cv::Point2d> pts,
                                                         bool isBigArmor) {
    cv::Mat Rmat, Tmat, R;
    Eigen::Matrix<double, 3, 1> xyz_camera;
    Eigen::Matrix<double, 3, 1> rvec_camera;
    Eigen::Matrix3d eigen_R;
    pnp_result result;
    if (isBigArmor)
        cv::solvePnP(BigArmor, pts, Kmat, Dmat, Rmat, Tmat, 0, cv::SOLVEPNP_IPPE);
    else
        cv::solvePnP(SmallArmor, pts, Kmat, Dmat, Rmat, Tmat, 0, cv::SOLVEPNP_IPPE);
    cv::cv2eigen(Tmat, xyz_camera);
    cv::cv2eigen(Rmat, rvec_camera);
    cv::Rodrigues(Rmat, R);
    cv::cv2eigen(R, eigen_R);
    result.xyz = trans("odom", detection_header.frame_id, xyz_camera);
    std::cout << "!!@@!!!" << std::endl;
    Eigen::Vector3d normal_word;
    normal_word << 0, 0, 1;
    result.normal_vec =
        trans("odom", detection_header.frame_id, eigen_R * normal_word + xyz_camera) - result.xyz;
    result.normal_vec =
        result.normal_vec / sqrt(pow(result.normal_vec[0], 2) + pow(result.normal_vec[1], 2) +
                                 pow(result.normal_vec[2], 2));
    result.show_vec = result.normal_vec * 0.2;
    return result;
}

cv::Point2d Position_Calculator::pos2img(Eigen::Matrix<double, 3, 1> X) {
    X = trans(detection_header.frame_id, "odom", X);
    X = K * X / X[2];
    return cv::Point2d(X[0], X[1]);
}

// std::vector<Eigen::Vector3d> Position_Calculator::pnp_get_pb_WM(
//     const std::vector<cv::Point2d> pts) {
//     cv::Mat rvec, tvec;
//     cv::Mat_<double> rotMat(3, 3);
//     Eigen::Matrix<double, 3, 3> R;
//     Eigen::Matrix<double, 3, 1> T;
//     std::vector<Eigen::Matrix<double, 3, 1>> pb;
//     cv::solvePnP(pw_energy, pts, Kmat, Dmat, rvec, tvec, 0, cv::SOLVEPNP_ITERATIVE);
//     if (rvec.size().height == 0 || rvec.size().width == 0)  // 防止误解情况直接 core dumped
//         return {};
//     rvec.convertTo(rvec, CV_64F);  // 旋转向量
//     tvec.convertTo(tvec, CV_64F);  // 平移向量
//     cv::Rodrigues(rvec, rotMat);
//     cv::cv2eigen(rotMat, R);
//     cv::cv2eigen(tvec, T);
//     for (int i = 0; i < 5; i++) {
//         Eigen::Matrix<double, 3, 1> p0;
//         p0 << pw_result[i][0], pw_result[i][1], pw_result[i][2];
//         pb.push_back(pc_to_pb(R * p0 + T));
//     }
//     pb.push_back(pc_to_pb(T));
//     return pb;
// }

