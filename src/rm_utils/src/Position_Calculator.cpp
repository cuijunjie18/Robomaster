#include <rm_utils/Position_Calculator.hpp>
// #include <sophus/se3.hpp>

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

Position_Calculator::pnp_result Position_Calculator::pnp(const std::vector<cv::Point2d> pts,
                                                         bool isBigArmor) {
    cv::Mat Rmat, Tmat, R;
    Eigen::Matrix<double, 3, 1> xyz_camera;
    Eigen::Matrix<double, 3, 1> rvec_camera;
    Eigen::Matrix3d eigen_R;
    pnp_result result;
    result.img_pts = pts;
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
    PerfGuard pnp("pnp_time");
    cv::Mat Rmat, Tmat, R;
    Eigen::Matrix<double, 3, 1> xyz_armor;
    Eigen::Matrix<double, 3, 1> rvec_camera;
    Eigen::Matrix3d eigen_R;
    pnp_result result;

    Eigen::Vector3d ground_normal_v, origin_v;
    ground_normal_v << 0, 0, 1;
    origin_v = Eigen::Vector3d::Zero();
    origin_v = trans(detection_header.frame_id, "odom", origin_v);
    ground_normal_v = trans(detection_header.frame_id, "odom", ground_normal_v) - origin_v;

    Eigen::Matrix<double, 9, 9> A;
    std::vector<cv::Vec3d> armors;
    if (isBigArmor) {
        armors = BigArmor;
    } else {
        armors = SmallArmor;
    }
    assert(pts.size() == armors.size());

    double x, y, fx, fy, cx, cy, u, v;
    fx = K(0, 0);
    fy = K(1, 1);
    cx = K(0, 2);
    cy = K(1, 2);

    /// @brief DLT PnP Solution
    for (int i = 0; i < 4; ++i) {
        x = armors[i][0];
        y = armors[i][1];
        u = pts[i].x;
        v = pts[i].y;
        A.row(2 * i) << x * fx, y * fx, fx, 0, 0, 0, x * cx - u * x, y * cx - u * y, cx - u;
        A.row(2 * i + 1) << 0, 0, 0, x * fy, y * fy, fy, x * cy - v * x, y * cy - v * y, cy - v;
    }
    /** @note
     * 去除坐标原点之间位置平移的影响 - origin_v
     * 1. odom系地面法向量、深度向量(仅限正对)同armro系x轴（即底边）垂直
     * odom->detection: trans([0,0,1]^T) -> ground_normal_v; trans([0,1,0]^T) -> depth_v;
     * armor->detection: R*[1,0,0]^T = r1
     * in detection: ground_normal_v & depth_v 垂直 r1
     * 2. armor<->detection x轴姿态不变(仅限正对)
     * in armor:     [0,1,0]^T & [0,0,1] 垂直 [1,0,0]^T // r1
     */
    A.row(8) << ground_normal_v[0], 0, 0, ground_normal_v[1], 0, 0, ground_normal_v[2], 0, 0;

    Eigen::JacobiSVD<Eigen::MatrixXd> svdA(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd singular_values = svdA.singularValues();
    int min_index;
    double min_value = singular_values.minCoeff(&min_index);
    Eigen::VectorXd X_ = svdA.matrixV().col(min_index);
    Eigen::Vector3d r0, r1, r2, t;
    r0 << X_[0], X_[3], X_[6];
    r1 << X_[1], X_[4], X_[7];
    t << X_[2], X_[5], X_[8];
    double beta = (1. / r0.norm() + 1. / r1.norm()) / 2;
    r0.normalize();
    r1.normalize();
    r2 = r0.cross(r1);  /// R矩阵正交基扩充
    eigen_R.col(0) = r0;
    eigen_R.col(1) = r1;
    eigen_R.col(2) = r2;
    if (beta * (x * X_[6] + y * X_[7] + X_[8]) < 0) {
        beta *= -1;
        eigen_R *= -1;
    }
    // t *= beta;

    //这里t算错了，暂时拿IPPE的t代替一下
    result.img_pts = pts;
    if (isBigArmor)
        cv::solvePnP(BigArmor, pts, Kmat, Dmat, Rmat, Tmat, 0, cv::SOLVEPNP_IPPE);
    else
        cv::solvePnP(SmallArmor, pts, Kmat, Dmat, Rmat, Tmat, 0, cv::SOLVEPNP_IPPE);
    cv::cv2eigen(Tmat, xyz_armor);

    // /// @brief IPPE
    // Eigen::Matrix<double, 3, 3> H; // 单应矩阵
    // H << eigen_R.block<3,2>(0,0), t;
    // H /= H(2,2);
    // Eigen::Matrix2d J;
    // Eigen::Vector2d centroid(0,0);
    // Eigen::Vector3d pv;
    // std::vector<Eigen::Vector2d> qs;
    // for (cv::Point2d pt : pts) {
    //     Eigen::Vector2d q;
    //     cv::cv2eigen(cv::Mat(pt), q);
    //     Eigen::Vector3d q_;
    //     q_ << q, 1;
    //     q_ = K.inverse() * q_;
    //     q = 1./q_[2] * q_.segment<2>(0);
    //     qs.push_back(q);
    // }
    // pv << centroid, 1;
    // pv = H * pv;
    // pv /= pv[2];

    // /// centroid = [0,0] 最优假设下
    // J << H(0,0) - H(2,0) * H(0,2), H(0,1) - H(2,1)*H(0,2), H(1,0) - H(2,0) * H(1,2), H(1,1) -
    // H(2,1) * H(1,2);

    // /// @brief cal Rv by Rodrigues
    // Eigen::Matrix3d Rv, axis_cross;
    // axis_cross << 0, 0, pv[0], 0, 0, pv[1], -pv[0], -pv[1], 0;
    // axis_cross /= sqrt(pv[0]*pv[0] + pv[1]*pv[1]);
    // double cos_t = 1./pv.norm(), sin_t = sqrt(1-cos_t*cos_t);
    // Rv = Eigen::Matrix3d::Identity() + sin_t*axis_cross + (1-cos_t)*axis_cross*axis_cross;

    // Eigen::Matrix<double, 3, 2> R32;
    // Eigen::Matrix<double, 2, 3> tmp;
    // Eigen::Matrix2d B, C, CTC, R22, bbT;
    // Eigen::Vector2d b, c;
    // Eigen::Vector3d r3, t1, t2;
    // Eigen::Matrix3d eigen_R2 = Eigen::Matrix3d::Zero();

    // tmp << Eigen::Matrix2d::Identity(), -pv.segment<2>(0);
    // tmp = tmp * Rv;
    // B = tmp.block<2,2>(0,0);
    // C = B.inverse()*J;
    // CTC = C.transpose() * C;
    // double depth = 1./sqrt(1./2 *
    // (CTC(0,0)+CTC(1,1)+sqrt((CTC(0,0)-CTC(1,1))*(CTC(0,0)-CTC(1,1))+4*CTC(0,1)*CTC(1,0)))); R22 =
    // depth * C; bbT = Eigen::Matrix2d::Identity() - R22.transpose() * R22; b << sqrt(bbT(0,0)),
    // -signbit(bbT(0,1))*sqrt(bbT(1,1)); R32 << R22, b.transpose(); R32.col(0).normalize();
    // R32.col(1).normalize();
    // r3 = R32.col(0).cross(R32.col(1));
    // /// 相机归一化平面视线[v,1]^T与armor z轴（转到detection坐标）夹角小于90度
    // if (pv.transpose() * r3 < 0) {
    //     r3 *= -1;
    // }
    // eigen_R << R32, r3;
    // /// 去除瑕旋转的点反演效果
    // if (eigen_R.determinant() <= 0) {
    //     eigen_R.col(2) *= -1;
    // }
    // eigen_R = Rv * eigen_R;

    // R32.row(2) *= -1;
    // r3.block<2,1>(0,0) *= -1;
    // eigen_R2 << R32, r3;
    // eigen_R2 = Rv * eigen_R2;
    // if (eigen_R2.determinant() <= 0) {
    //     eigen_R2.col(2) *= -1;
    // }

    // t1 = depth * pv - eigen_R.block<3,2>(0,0) * centroid;
    // t2 = depth * pv - eigen_R2.block<3,2>(0,0) * centroid;

    // /// @brief 最小二乘解t
    // Eigen::MatrixXd W(2 * qs.size(), 3);
    // Eigen::VectorXd s1(2 * qs.size(), 1), s2(2 * qs.size(), 1);
    // for (int i = 0; i < qs.size(); ++i) {
    //     Eigen::Vector3d armor;
    //     cv::cv2eigen(armors[i], armor);
    //     W.row(2 * i) << 1, 0, -qs[i][0];
    //     W.row(2 * i + 1) << 0, 1, -qs[i][1];
    //     s1.block<2, 1>(2 * i, 0) << eigen_R.row(2) * armor * qs[i] -
    //                                     eigen_R.block<2, 2>(0, 0) * armor.segment<2>(0);
    //     s2.block<2, 1>(2 * i, 0) << eigen_R2.row(2) * armor * qs[i] -
    //                                     eigen_R2.block<2, 2>(0, 0) * armor.segment<2>(0);
    // }

    // /** @brief SVD代替W^T*W求逆
    //  * (W.transpose()*W)行列式过小时 t = (W.transpose()*W).inverse()*W.transpose()*s2; 求解无果
    //  */
    // Eigen::JacobiSVD<Eigen::MatrixXd> svdW(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // Eigen::MatrixXd S(2 * qs.size(), 3);
    // S.setZero();
    // for (int i = 0; i < svdW.singularValues().size(); ++i) {
    //     S(i, i) = svdW.singularValues()[i];
    // }

    // /// 两个矩阵之间有z轴取反的效果，根据投影误差取最小者
    // if ((W * t1 - s1).norm() > (W * t2 - s2).norm()) {
    //     eigen_R = eigen_R2;
    //     t = svdW.matrixV() * (S.transpose() * S).inverse() * S.transpose() *
    //         svdW.matrixU().transpose() * s2;
    // } else {
    //     t = svdW.matrixV() * (S.transpose() * S).inverse() * S.transpose() *
    //         svdW.matrixU().transpose() * s1;
    // }

    // /// 迭代PnP
    // double cost = 0, last_cost;
    // int iterations = 20;
    // if (!Sophus::isOrthogonal(eigen_R)) {
    //     double angle = acos((eigen_R.trace() - 1)/2);
    //     double x = (eigen_R(2,1) - eigen_R(1,2))/sqrt((eigen_R(2,1) - eigen_R(1,2))*(eigen_R(2,1)
    //     - eigen_R(1,2))+(eigen_R(0,2) - eigen_R(2,0))*(eigen_R(0,2) - eigen_R(2,0))+(eigen_R(1,0)
    //     - eigen_R(0,1))*(eigen_R(1,0) - eigen_R(0,1))); double y = (eigen_R(0,2) -
    //     eigen_R(2,0))/sqrt((eigen_R(2,1) - eigen_R(1,2))*(eigen_R(2,1) -
    //     eigen_R(1,2))+(eigen_R(0,2) - eigen_R(2,0))*(eigen_R(0,2) - eigen_R(2,0))+(eigen_R(1,0) -
    //     eigen_R(0,1))*(eigen_R(1,0) - eigen_R(0,1))); double z = (eigen_R(1,0) -
    //     eigen_R(0,1))/sqrt((eigen_R(2,1) - eigen_R(1,2))*(eigen_R(2,1) -
    //     eigen_R(1,2))+(eigen_R(0,2) - eigen_R(2,0))*(eigen_R(0,2) - eigen_R(2,0))+(eigen_R(1,0) -
    //     eigen_R(0,1))*(eigen_R(1,0) - eigen_R(0,1))); eigen_R = Eigen::AngleAxisd(angle,
    //     Eigen::Vector3d(x, y, z)).toRotationMatrix();
    // }
    // std::cout << "IsOrthogonal: " << Sophus::isOrthogonal(eigen_R) << std::endl;
    // Sophus::SE3d pose(eigen_R, t);
    // Eigen::Matrix<double, 6, 1> dx = Eigen::Matrix<double, 6, 1>::Zero();
    // for (int iter = 0; iter < iterations; ++iter) {
    //     Eigen::Matrix<double, 6, 6> He = Eigen::Matrix<double, 6, 6>::Zero();
    //     Eigen::Matrix<double, 6, 1> be = Eigen::Matrix<double, 6, 1>::Zero();
    //     for (int i = 0; i < pts.size(); ++i) {
    //         Eigen::Vector2d pt;
    //         Eigen::Vector3d armor;
    //         cv::cv2eigen(cv::Mat(pts[i]), pt);
    //         cv::cv2eigen(cv::Mat(armors[i]), armor);
    //         armor = pose * armor;
    //         double inv_z = 1./armor[2], inv_z2 = inv_z * inv_z;
    //         Eigen::Vector2d e = pt - inv_z * (K * armor).segment<2>(0);
    //         cost += e.squaredNorm();
    //         Eigen::Matrix<double, 2, 6> Je;
    //         Je << -fx * inv_z, 0,  //
    //             fx * armor[0] * inv_z2, fx * armor[0] * armor[1] * inv_z2, //
    //             -fx - fx * armor[0] * armor[0] * inv_z2, fx * armor[1] * inv_z,
    //             0, -fy * inv_z,
    //             fy * armor[1] * inv_z, fy + fy * armor[1] * armor[1] * inv_z2,
    //             -fy * armor[0] * armor[1] * inv_z2, -fy * armor[0] * inv_z;
    //         He += Je.transpose() * Je;
    //         be += -Je.transpose() * e;
    //     }
    //     dx = He.ldlt().solve(be);
    //     if (isnan(dx[0])) {
    //         std::cout << "result is nan!" << std::endl;
    //         break;
    //     }
    //     // 判断是否发散
    //     if (iter > 0 && cost >= last_cost) {
    //         std::cout << "cost: " << cost << ", last cost: " << last_cost << std::endl;
    //         break;
    //     }
    //     last_cost = cost;
    //     pose = Sophus::SE3d::exp(dx) * pose;
    //     // judge if converge
    //     if (dx.norm() < 1e-6) {
    //         std::cout << "[Converge] " << "iter times: " << iter << "dx: " << dx << "cost: " <<
    //         cost << std::endl; break;
    //     }
    // }
    // std::cout << "dx: " << dx << "norm: " << dx.norm() << std::endl;

    // Eigen::Matrix4d T = pose.matrix();
    // eigen_R = T.block<3, 3>(0, 0);
    // t = T.block<3, 1>(0, 3);

    // xyz_armor = t;
    result.img_pts = pts;

    result.xyz = trans("odom", detection_header.frame_id, xyz_armor);
    Eigen::Vector3d normal_armor;
    normal_armor << 0, 0, 1;
    result.normal_vec =
        trans("odom", detection_header.frame_id, eigen_R * normal_armor + xyz_armor) -
        result.xyz;  // 坐标变换到odom系并将起点平移至odom系原点

    result.normal_vec.normalize();
    result.show_vec = result.normal_vec * 0.2;
    result.yaw = atan2(result.normal_vec[1], result.normal_vec[0]);
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
