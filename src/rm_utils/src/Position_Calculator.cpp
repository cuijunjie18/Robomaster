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
    // PerfGuard trans_perf_guard("trans_Total");
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
Eigen::Vector3d Position_Calculator::generate_armor_point_odom(double pitch, double yaw,
                                                               Eigen::Vector3d xyz,
                                                               Eigen::Vector3d point_armor) {
    Eigen::Vector3d odom_point = point_armor;
    double pitch_rad = pitch / 180 * M_PI;
    double yaw_rad = yaw / 180 * M_PI;
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
    return odom_point;
}
std::vector<cv::Point2d> Position_Calculator::generate_armor_img(bool isBigArmor, double pitch,
                                                                 double yaw, Eigen::Vector3d xyz) {
    std::vector<cv::Point2d> armor_img;
    std::vector<cv::Vec3d> armor_pts;
    if (isBigArmor) {
        armor_pts = BigArmor;
    } else {
        armor_pts = SmallArmor;
    }
    for (int i = 0; i < armor_pts.size(); ++i) {
        // armor系转odom系
        Eigen::Vector3d odom_point;
        cv::cv2eigen(armor_pts[i], odom_point);
        odom_point = generate_armor_point_odom(pitch, yaw, xyz, odom_point);
        armor_img.push_back(pos2img(odom_point));
    }
    armor_img.push_back(pos2img(xyz));
    return armor_img;
}

double Position_Calculator::diff_fun_nor_dis(std::vector<cv::Point2d> img_pts, Eigen::Vector3d xyz,
                                             std::vector<cv::Point2d> guess_pts) {
    double diff = 0.0;
    cv::Point2d img_center = (img_pts[0] + img_pts[1] + img_pts[2] + img_pts[3]) / 4.0;
    cv::Point2d guess_centre = pos2img(xyz);
    cv::Point2d diff_vec = img_center - guess_centre;
    for (int i = 0; i < img_pts.size(); ++i) {
        img_pts[i] += diff_vec;
        diff += sqrt((img_pts[i] - guess_pts[i]).ddot(img_pts[i] - guess_pts[i]));
    }
    return diff;
}

double Position_Calculator::diff_fun_side_angle(std::vector<cv::Point2d> img_pts,
                                                Eigen::Vector3d xyz,
                                                std::vector<cv::Point2d> guess_pts) {
    double diff = 0.0;
    for (int i = 0; i < img_pts.size(); ++i) {
        cv::Point2d vec1, vec2;
        vec1 = guess_pts[i] - guess_pts[(i + 1) % img_pts.size()];
        vec2 = img_pts[i] - img_pts[(i + 1) % img_pts.size()];
        diff += acos(vec1.ddot(vec2) / sqrt(vec1.ddot(vec1) * vec2.ddot(vec2)));
    }
    return diff;
}

double Position_Calculator::diff_fun_area(std::vector<cv::Point2d> img_pts, Eigen::Vector3d xyz,
                                          std::vector<cv::Point2d> guess_pts) {
    double diff = 0.0;
    std::vector<cv::Point2d> diff_img_pts(4), diff_guess_pts(4);
    // std::cout << "AAAAAA" << std::endl;
    diff_img_pts[0] = img_pts[1] - img_pts[0];
    diff_img_pts[1] = img_pts[3] - img_pts[0];
    diff_img_pts[2] = img_pts[2] - img_pts[1];
    diff_img_pts[3] = img_pts[2] - img_pts[3];
    // std::cout << "BBB" << std::endl;

    diff_guess_pts[0] = guess_pts[1] - guess_pts[0];
    diff_guess_pts[1] = guess_pts[3] - guess_pts[0];
    diff_guess_pts[2] = guess_pts[2] - guess_pts[1];
    diff_guess_pts[3] = guess_pts[2] - guess_pts[3];
    // std::cout << "CCC" << std::endl;

    double S_img_pts, S_guess_pts;
    S_img_pts = abs(diff_img_pts[0].x * diff_img_pts[1].y - diff_img_pts[0].y * diff_img_pts[1].x) +
                abs(diff_img_pts[2].x * diff_img_pts[3].y - diff_img_pts[2].y * diff_img_pts[3].x);
    S_guess_pts =
        abs(diff_guess_pts[0].x * diff_guess_pts[1].y - diff_guess_pts[0].y * diff_guess_pts[1].x) +
        abs(diff_guess_pts[2].x * diff_guess_pts[3].y - diff_guess_pts[2].y * diff_guess_pts[3].x);

    diff = abs(S_img_pts / S_guess_pts - 1.0);
    return diff;
}

double Position_Calculator::diff_fun_left_right_ratio(std::vector<cv::Point2d> img_pts,
                                                      Eigen::Vector3d xyz,
                                                      std::vector<cv::Point2d> guess_pts) {
    double diff = 0.0;
    double propotion_img, propotion_guess;
    propotion_img = (img_pts[0] - img_pts[1]).ddot(img_pts[0] - img_pts[1]) /
                    (img_pts[2] - img_pts[3]).ddot(img_pts[2] - img_pts[3]);
    propotion_guess = (guess_pts[0] - guess_pts[1]).ddot(guess_pts[0] - guess_pts[1]) /
                      (guess_pts[2] - guess_pts[3]).ddot(guess_pts[2] - guess_pts[3]);
    diff = abs(propotion_guess - propotion_img);
    return diff;
}

double Position_Calculator::final_diff_fun_cal(bool isBigArmor, std::vector<cv::Point2d> img_pts,
                                               Eigen::Vector3d xyz, double pitch, double yaw) {
    double diff = 0.0;
    std::vector<cv::Point2d> guess_pts = generate_armor_img(isBigArmor, pitch, yaw, xyz);
    std::vector<double> param = {1.0, -10.0, 10.0, -4.61003499};
    diff = param[0] * diff_fun_nor_dis(img_pts, xyz, guess_pts) +
           param[1] * diff_fun_side_angle(img_pts, xyz, guess_pts) +
           param[2] * diff_fun_area(img_pts, xyz, guess_pts) +
           param[3] * diff_fun_left_right_ratio(img_pts, xyz, guess_pts);
    return diff;
}

double Position_Calculator::final_diff_fun_choose(bool isBigArmor, std::vector<cv::Point2d> img_pts,
                                                  Eigen::Vector3d xyz, double pitch, double yaw) {
    double diff = 0.0;
    std::vector<cv::Point2d> guess_pts = generate_armor_img(isBigArmor, pitch, yaw, xyz);
    std::vector<double> param = {1.0, 27.99950239, -10.0, -9.74106768};
    diff = param[0] * diff_fun_nor_dis(img_pts, xyz, guess_pts) +
           param[1] * diff_fun_side_angle(img_pts, xyz, guess_pts) +
           param[2] * diff_fun_area(img_pts, xyz, guess_pts) +
           param[3] * diff_fun_left_right_ratio(img_pts, xyz, guess_pts);
    return diff;
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
    Eigen::Vector3d normal_word;
    normal_word << 0, 0, 1;
    result.normal_vec =
        trans("odom", detection_header.frame_id, eigen_R * normal_word + xyz_camera) - result.xyz;
    result.normal_vec =
        result.normal_vec / sqrt(pow(result.normal_vec[0], 2) + pow(result.normal_vec[1], 2) +
                                 pow(result.normal_vec[2], 2));
    result.show_vec = result.normal_vec * 0.2;
    result.yaw = atan2(result.normal_vec[1], result.normal_vec[0]);
    return result;
}

Position_Calculator::pnp_result Position_Calculator::rm_pnp(const std::vector<cv::Point2d> pts,
                                                            bool isBigArmor) {
    PerfGuard PNP_perf_guard("PNP_Total");

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
    PerfGuard PNP_yaw_perf_guard("PNP_get_yaw");
    // 三分法求解
    double pitch = -15.0;
    int iter_num = 0;
    double mid_angle, l1, r1, mid1, l2, r2, mid2, eps;
    mid_angle = atan(result.xyz[1] / result.xyz[0]) / M_PI * 180.0;
    l1 = mid_angle - 70.0;
    r1 = mid_angle;
    l2 = mid_angle;
    r2 = mid_angle + 70.0;
    eps = 2.0;
    while ((r1 - l1 > eps) || (r2 - l2 > eps)) {
        mid1 = (r1 + l1) / 2;
        if (final_diff_fun_cal(isBigArmor, pts, result.xyz, pitch, mid1 - eps) >
            final_diff_fun_cal(isBigArmor, pts, result.xyz, pitch, mid1 + eps)) {
            l1 = mid1;
        } else {
            r1 = mid1;
        }

        mid2 = (r2 + l2) / 2;
        if (final_diff_fun_cal(isBigArmor, pts, result.xyz, pitch, mid2 - eps) >
            final_diff_fun_cal(isBigArmor, pts, result.xyz, pitch, mid2 + eps)) {
            l2 = mid2;
        } else {
            r2 = mid2;
        }
        ++iter_num;
        // if (final_diff_fun_cal(isBigArmor, pts, result.xyz, pitch, (2 * l1 + r1) / 3) <
        //     final_diff_fun_cal(isBigArmor, pts, result.xyz, pitch, (l1 + 2 * r1) / 3)) {
        //     r1 = (l1 + 2 * r1) / 3;
        // } else {
        //     l1 = (2 * l1 + r1) / 3;
        // }
        // if (final_diff_fun_cal(isBigArmor, pts, result.xyz, pitch, (2 * l2 + r2) / 3) <
        //     final_diff_fun_cal(isBigArmor, pts, result.xyz, pitch, (l2 + 2 * r2) / 3)) {
        //     r2 = (l2 + 2 * r2) / 3;
        // } else {
        //     l2 = (2 * l2 + r2) / 3;
        // }
    }
    // std::cout << "fps   r1 - l1: " << abs(r1 - l1) << "   r2 - l2: " << abs(r2 - l2)
    //           << "  iter_num: " << iter_num << std::endl;
    double yaw = 0;
    double yaw_1 = (l1 + r1) / 2;
    double yaw_2 = (l2 + r2) / 2;
    if (final_diff_fun_choose(isBigArmor, pts, result.xyz, pitch, yaw_1) <
        final_diff_fun_choose(isBigArmor, pts, result.xyz, pitch, yaw_2)) {
        yaw = yaw_1;
    } else {
        yaw = yaw_2;
    }
    Eigen::Vector3d normal_armor;
    normal_armor << 0, 0, 1;
    result.normal_vec =
        generate_armor_point_odom(pitch, yaw, result.xyz, normal_armor) - result.xyz;
    result.normal_vec =
        result.normal_vec / sqrt(pow(result.normal_vec[0], 2) + pow(result.normal_vec[1], 2) +
                                 pow(result.normal_vec[2], 2));
    result.show_vec = result.normal_vec * 0.2;
    result.yaw = atan2(result.normal_vec[1], result.normal_vec[0]);
    // std::cout << "error11: " << result.yaw * 180 / M_PI - (l1 + r1) / 2 << std::endl;
    // std::cout << "error22: " << result.yaw * 180 / M_PI - (l2 + r2) / 2 << std::endl;
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
