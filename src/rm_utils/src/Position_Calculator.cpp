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

void Position_Calculator::update_trans(const Eigen::Matrix<double, 4, 4>& trans_) {
    trans = trans_;
}

void Position_Calculator::update_camera_info(const std::vector<double>& k_,
                                             const std::vector<double>& d_) {
    assert(k_.size() == 9 && "Camera Matrix must has 9 elements.");
    assert(d_.size() == 5 && "Distortion Matrix must has 9 elements.");
    K = Eigen::Matrix<double, 3, 3>(k_.data()).transpose();
    D = Eigen::Matrix<double, 1, 5>(d_.data());
    cv::eigen2cv(K, Kmat);
    cv::eigen2cv(D, Dmat);
}

Eigen::Matrix<double, 3, 1> Position_Calculator::pb_to_pc(Eigen::Matrix<double, 3, 1> pb) {
    // return cam2pitch_R.inverse() * (pitch2yaw_R.inverse() * (yaw2base_R.inverse() * pb -
    // pitch2yaw_T) - cam2pitch_T);
    Eigen::Matrix<double, 4, 1> pc;
    pc << pb[0], pb[1], pb[2], 1.;
    pc = trans.inverse() * pc;
    return pc.block<3, 1>(0, 0);
}
Eigen::Matrix<double, 3, 1> Position_Calculator::pc_to_pb(Eigen::Matrix<double, 3, 1> pc) {
    // return yaw2base_R * (pitch2yaw_R * (cam2pitch_R * pc + cam2pitch_T) + pitch2yaw_T);
    Eigen::Matrix<double, 4, 1> pb;
    pb << pc[0], pc[1], pc[2], 1.;
    pb = trans * pb;
    return pb.block<3, 1>(0, 0);
}

Eigen::Matrix<double, 3, 1> Position_Calculator::pnp_get_pb(const std::vector<cv::Point2d> pts,
                                                            bool isBigArmor) {
    cv::Mat Rmat, Tmat;
    Eigen::Matrix<double, 3, 1> pc;
    if (isBigArmor)
        cv::solvePnP(BigArmor, pts, Kmat, Dmat, Rmat, Tmat, 0, cv::SOLVEPNP_IPPE);
    else
        cv::solvePnP(SmallArmor, pts, Kmat, Dmat, Rmat, Tmat, 0, cv::SOLVEPNP_IPPE);
    cv::cv2eigen(Tmat, pc);
    return pc_to_pb(pc);
}

Position_Calculator::pnp_result Position_Calculator::pnp(const std::vector<cv::Point2d> pts,
                                                         bool isBigArmor) {
    cv::Mat Rmat, Tmat, R;
    Eigen::Matrix<double, 3, 1> pc;
    Eigen::Matrix<double, 3, 1> rvec;
    Eigen::Matrix3d eigen_R;
    pnp_result result;
    if (isBigArmor)
        cv::solvePnP(BigArmor, pts, Kmat, Dmat, Rmat, Tmat, 0, cv::SOLVEPNP_IPPE);
    else
        cv::solvePnP(SmallArmor, pts, Kmat, Dmat, Rmat, Tmat, 0, cv::SOLVEPNP_IPPE);
    cv::cv2eigen(Tmat, pc);
    cv::cv2eigen(Rmat, rvec);
    cv::Rodrigues(Rmat, R);
    cv::cv2eigen(R, eigen_R);
    result.xyz = pc_to_pb(pc);
    // result.Rvec = rvec;

    Eigen::Vector3d normal_word;
    normal_word << 0, 0, 1;
    result.normal_vec = pc_to_pb(eigen_R * normal_word + pc) - result.xyz;
    result.normal_vec =
        result.normal_vec / sqrt(pow(result.normal_vec[0], 2) + pow(result.normal_vec[1], 2) +
                                 pow(result.normal_vec[2], 2));
    result.show_vec = result.normal_vec * 0.2;
    return result;
}

std::vector<Eigen::Vector3d> Position_Calculator::pnp_get_pb_WM(
    const std::vector<cv::Point2d> pts) {
    cv::Mat rvec, tvec;
    cv::Mat_<double> rotMat(3, 3);
    Eigen::Matrix<double, 3, 3> R;
    Eigen::Matrix<double, 3, 1> T;
    std::vector<Eigen::Matrix<double, 3, 1>> pb;
    cv::solvePnP(pw_energy, pts, Kmat, Dmat, rvec, tvec, 0, cv::SOLVEPNP_ITERATIVE);
    if (rvec.size().height == 0 || rvec.size().width == 0)  // 防止误解情况直接 core dumped
        return {};
    rvec.convertTo(rvec, CV_64F);  // 旋转向量
    tvec.convertTo(tvec, CV_64F);  // 平移向量
    cv::Rodrigues(rvec, rotMat);
    cv::cv2eigen(rotMat, R);
    cv::cv2eigen(tvec, T);
    for (int i = 0; i < 5; i++) {
        // Pc=R*Po+T
        Eigen::Matrix<double, 3, 1> p0;
        p0 << pw_result[i][0], pw_result[i][1], pw_result[i][2];
        pb.push_back(pc_to_pb(R * p0 + T));
    }
    pb.push_back(pc_to_pb(T));
    return pb;
}

cv::Point2d Position_Calculator::pos2img(Eigen::Matrix<double, 3, 1> X) {
    X = pb_to_pc(X);
    X = K * X / X[2];
    return cv::Point2d(X[0], X[1]);
}