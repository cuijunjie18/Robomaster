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
    Eigen::Matrix<double, 3, 1> xyz_camera;
    Eigen::Matrix<double, 3, 1> rvec_camera;
    Eigen::Matrix3d eigen_R;
    pnp_result result;

    Eigen::Vector3d ground_normal_v, depth_normal_v, origin_v;
    ground_normal_v << 0, 0, 1;
    depth_normal_v << 0, 1, 0; // odom z轴向上？
    origin_v = Eigen::Vector3d::Zero();
    origin_v = trans(detection_header.frame_id, "odom", origin_v);
    ground_normal_v = trans(detection_header.frame_id, "odom", ground_normal_v) - origin_v;
    depth_normal_v = trans(detection_header.frame_id, "odom", depth_normal_v) - origin_v;
    
    Eigen::Matrix<double, 13 , 9> A;  
    std::vector<cv::Vec3d> armors;
    if (isBigArmor) {
        armors = BigArmor;
    } else {
        armors = SmallArmor;
    }
    assert(pts.size()==armors.size());

    double x, y, fx, fy, cx, cy, u, v;
    fx = K(0, 0);
    fy = K(1, 1);
    cx = K(0, 2);
    cy = K(1, 2);

    // DLT PnP Solution
    for (int i = 0; i < 4; ++i) {
        x = armors[i][0];
        y = armors[i][1];
        u = pts[i].x;
        v = pts[i].y;
        A.row(2 * i) << x * fx, y * fx, fx, 0, 0, 0, x * cx - u * x, y * cx - u * y, cx - u;
        A.row(2 * i + 1) << 0, 0, 0, x * fy, y * fy, fy, x * cy - v * x, y * cy - v * y, cy - v;
    }
    // armor<->odom x轴姿态不变, 地面法向量、深度向量同各自x轴垂直
    A.row(8) << 0,0,0,0,0,0,1,0,0; // odom[0,0,1]^T垂直r1
    A.row(9) << 0,0,0,1,0,0,0,0,0; // odom[0,1,0]^T垂直r1
    A.row(10) << origin_v[0], 0, 0, origin_v[1], 0, 0, origin_v[2], 0, 0;
    A.row(11) << depth_normal_v[0], 0, 0, depth_normal_v[1], 0, 0, depth_normal_v[2], 0, 0; 
    A.row(12) << ground_normal_v[0], 0, 0, ground_normal_v[1], 0, 0, ground_normal_v[2], 0, 0;
    
    Eigen::JacobiSVD<Eigen::MatrixXd> svdA(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd singular_values = svdA.singularValues();
    int min_index;
    double min_value = singular_values.minCoeff(&min_index);
    Eigen::VectorXd X_ = svdA.matrixV().col(min_index);
    Eigen::Vector3d r0, r1, r2, t;
    r0 << X_[0], X_[3], X_[6];
    r1 << X_[1], X_[4], X_[7];
    t << X_[2], X_[5], X_[8];
    double beta = (1./r0.norm() + 1./r1.norm()) / 2;
    r0.normalize();
    r1.normalize();
    r2 = r0.cross(r1);  // R矩阵扩充正交基
    eigen_R.col(0) = r0;
    eigen_R.col(1) = r1;
    eigen_R.col(2) = r2;
    // Eigen::JacobiSVD<Eigen::MatrixXd> svdR(eigen_R, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // eigen_R = svdR.matrixU() * (svdR.matrixV().transpose());
    // double beta = 3 / (svdR.singularValues().sum());
    // double beta = pow(eigen_R.determinant(),-1./3);
    if (beta * (x * X_[6] + y * X_[7] + X_[8]) < 0) {
        beta *= -1;
        eigen_R *= -1;
    }
    t *= beta;

    // cv::solvePnP(armor, pts, Kmat, Dmat, Rmat, Tmat, 0, cv::SOLVEPNP_IPPE);

    // IPPE
    Eigen::Matrix<double, 3, 3> H; // 单应矩阵
    H << eigen_R.block<3,2>(0,0), t;
    H /= H(2,2);
    Eigen::Matrix3d H_adj = H.adjoint();
    Eigen::Matrix2d J;
    Eigen::Vector2d centroid(0,0); 
    Eigen::Vector3d pv;
    std::vector<Eigen::Vector2d> qs;
    for (cv::Point2d pt : pts) {
        Eigen::Vector2d q;
        cv::cv2eigen(cv::Mat(pt), q);
        Eigen::Vector3d q_;
        q_ << q, 1;
        q_ = K.inverse() * q_;
        q = 1./q_[2] * q_.segment<2>(0);
        qs.push_back(q);
    }
    for (cv::Vec3d & armor: armors) {
        Eigen::Vector3d armor_eigen;
        cv::cv2eigen(armor,armor_eigen);
        centroid += armor_eigen.segment<2>(0);
    }
    centroid /= armors.size();
    // centroid = [0,0] ?
    // pv = trans("odom", detection_header.frame_id, normal_world);
    // pv = K.inverse() * pts.centroid();
    pv << centroid, 1;
    pv = H * pv;
    pv /= pv[2];

    J << H_adj(1,1) - centroid[1]*H_adj(2,1), -H_adj(0,1)+centroid[0]*H_adj(2,1), -H_adj(1,0) + centroid[1]*H_adj(2,0), H_adj(0,0) - centroid[0]*H_adj(2,0);
    J /= pow((centroid[0]*H(2,0)+centroid[1]*H(2,1)+H(2,2)),-2);

    // cal Rv by Rodrigues
    Eigen::Matrix3d Rv, axis_cross;
    axis_cross << 0, 0, pv[0], 0, 0, pv[1], -pv[0], -pv[1], 0;
    axis_cross *= sqrt(pv[0]*pv[0] + pv[1]*pv[1]);
    double cost = 1./pv.norm(), sint = sqrt(1-cost*cost);
    Rv = Eigen::Matrix3d::Identity() + sint*axis_cross + (1-cost)*axis_cross*axis_cross;

    Eigen::Matrix<double, 3, 2> R32;
    Eigen::Matrix<double, 2, 3> tmp;
    Eigen::Matrix2d B, C, CCT, R22, bbT;
    Eigen::Vector2d b, c;
    Eigen::Vector3d r3;

    tmp << Eigen::Matrix2d::Identity(), -pv.segment<2>(0);
    tmp = tmp * Rv;
    B = tmp.block<2,2>(0,0);
    C = B.inverse()*J;
    CCT = C*C.transpose();
    double inv_depth = sqrt(1./2 * (CCT(0,0)+CCT(1,1)+sqrt((CCT(0,0)-CCT(1,1))*(CCT(0,0)-CCT(1,1))+4*CCT(0,1)*CCT(1,0))));
    R22 = inv_depth * C;
    bbT = Eigen::Matrix2d::Identity() - R22.transpose() * R22;
    b << sqrt(bbT(0,0)), -signbit(bbT(0,1))*sqrt(bbT(1,1));
    R32 << R22, b.transpose();
    r3 = R32.col(0).cross(R32.col(1));
    if (pv.transpose() * r3 < 0) {
        R32.row(2) *= -1;
        r3.block<2,1>(0,0) *= -1;
    }
    // eigen_R << R32, r3;
    // eigen_R = Rv * eigen_R;

    // 最小二乘解t
    Eigen::MatrixXd W(2*qs.size(), 3);
    Eigen::VectorXd s(2*qs.size(), 1);
    for (int i = 0; i < qs.size(); ++i) {
        Eigen::Vector3d armor;
        cv::cv2eigen(armors[i], armor);
        W.row(2*i) << 1, 0, -qs[i][0];
        W.row(2*i+1) << 0, 1, -qs[i][1];
        s.block<2,1>(2*i,0) << eigen_R.row(2)*armor*qs[i] - eigen_R.block<2,2>(0,0)*armor.segment<2>(0);
    }
    // t = (W.transpose()*W).inverse()*W.transpose()*s;

    xyz_camera = t;  // camera in world's coordinates;
    result.img_pts = pts;

    result.xyz = trans("odom", detection_header.frame_id, xyz_camera);
    Eigen::Vector3d normal_world;
    normal_world << 0, 0, 1;  // world(armor)系下装甲板法向量
    result.normal_vec =
        trans("odom", detection_header.frame_id, eigen_R * normal_world + xyz_camera) - result.xyz; // 坐标变换到odom系并将起点平移至odom系原点
    // result.normal_vec = trans("odom", detection_header.frame_id, normal_world) - result.xyz;
    result.normal_vec = result.normal_vec.normalized();
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
