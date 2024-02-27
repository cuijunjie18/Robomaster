#include "enemy_predictor/enemy_kf.hpp"
enemy_KF_4::enemy_KF_4(rclcpp::Node *node_) : logger(rclcpp::get_logger("enemy_EKF")), node(node_) {
    sample_num = 2 * state_num;
    samples = std::vector<Vn>(sample_num);
    weights = std::vector<double>(sample_num);
    Pe = init_P.asDiagonal();
    sample_X = std::vector<Vn>(sample_num);
    load_params();
}

enemy_KF_4::Vn enemy_KF_4::get_X(State _state) const {
    Vn _X;
    _X[0] = _state.x;
    _X[1] = _state.vx;
    _X[2] = _state.y;
    _X[3] = _state.vy;
    _X[4] = _state.yaw;
    _X[5] = _state.omega;
    return _X;
}

enemy_KF_4::State enemy_KF_4::get_state(Vn _X) const {
    State _state;
    _state.x = _X[0];
    _state.vx = _X[1];
    _state.y = _X[2];
    _state.vy = _X[3];
    _state.yaw = _X[4];
    _state.omega = _X[5];
    return _state;
}

enemy_KF_4::Vm enemy_KF_4::get_Z(Output _output) {
    Vm result;
    result[0] = _output.x;
    result[1] = _output.y;
    result[2] = _output.z;
    result[3] = _output.yaw;
    return result;
}

enemy_KF_4::Vm2 enemy_KF_4::get_Z(Output2 _output) {
    Vm2 result;
    result[0] = _output.x;
    result[1] = _output.y;
    result[2] = _output.z;
    result[3] = _output.yaw;
    result[4] = _output.x2;
    result[5] = _output.y2;
    result[6] = _output.z2;
    result[7] = _output.yaw2;
    return result;
}

enemy_KF_4::Output enemy_KF_4::get_output(Vm _Z) {
    Output result;
    result.x = _Z[0];
    result.y = _Z[1];
    result.z = _Z[2];
    result.yaw = _Z[3];
    return result;
}

enemy_KF_4::Output2 enemy_KF_4::get_output(Vm2 _Z) {
    Output2 result;
    result.x = _Z[0];
    result.y = _Z[1];
    result.z = _Z[2];
    result.yaw = _Z[3];
    result.x2 = _Z[4];
    result.y2 = _Z[5];
    result.z2 = _Z[6];
    result.yaw2 = _Z[7];
    return result;
}

void enemy_KF_4::reset(const Output &observe, int phase_id, int armor_cnt_, double stamp) {
    std::vector<double> dis(armor_cnt_, 0.2);
    std::vector<double> z(armor_cnt_, -0.1);
    state = State(observe.x, 0, observe.y, 0, 0, 0);
    state.yaw = observe.yaw;
    Xe = get_X(state);
    Pe = init_P.asDiagonal();
    const_dis = dis;
    const_z = z;
    armor_cnt = armor_cnt_;
    angle_dis = 2 * M_PI / armor_cnt_;
    timestamp = stamp;
}

enemy_KF_4::Vn enemy_KF_4::f(const Vn &X, double dT) const {
    State X_state = get_state(X);
    X_state.x = X_state.x + X_state.vx * dT;
    X_state.y = X_state.y + X_state.vy * dT;
    X_state.yaw = X_state.yaw + X_state.omega * dT;
    Vn result = get_X(X_state);
    return result;
}
enemy_KF_4::Vm enemy_KF_4::h(const Vn &X, int phase_id) {
    State X_state = get_state(X);
    Output Z_output;
    Z_output.yaw = X_state.yaw;
    Z_output.x = X_state.x + const_dis[phase_id] * cos(X_state.yaw + phase_id * angle_dis);
    Z_output.y = X_state.y + const_dis[phase_id] * sin(X_state.yaw + phase_id * angle_dis);
    Z_output.z = const_z[phase_id];
    Vm result = get_Z(Z_output);
    return result;
}

enemy_KF_4::Vm2 enemy_KF_4::h(const Vn &X, int phase_id, int phase_id2) {
    State X_state = get_state(X);
    Output2 Z_output;
    Z_output.yaw = X_state.yaw;
    Z_output.x = X_state.x + const_dis[phase_id] * cos(X_state.yaw + phase_id * angle_dis);
    Z_output.y = X_state.y + const_dis[phase_id] * sin(X_state.yaw + phase_id * angle_dis);
    Z_output.z = const_z[phase_id];
    Z_output.yaw2 = X_state.yaw;
    Z_output.x2 = X_state.x + const_dis[phase_id2] * cos(X_state.yaw + phase_id2 * angle_dis);
    Z_output.y2 = X_state.y + const_dis[phase_id2] * sin(X_state.yaw + phase_id2 * angle_dis);
    Z_output.z2 = const_z[phase_id2];
    Vm2 result = get_Z(Z_output);
    return result;
}

void enemy_KF_4::SRCR_sampling_3(Vn _x, Mnn _P)  // 3阶球面——径向采样法
{
    double sqrtn = sqrt(state_num);
    double weight = 1.0 / (2 * state_num);
    Eigen::LLT<Eigen::MatrixXd> get_S(_P);
    Eigen::MatrixXd S = get_S.matrixL();
    for (int i = 0; i < state_num; ++i) {
        samples[i] = _x + sqrtn * S.col(i);

        weights[i] = weight;

        samples[i + state_num] = _x - sqrtn * S.col(i);
        weights[i + state_num] = weight;
    }
}

void enemy_KF_4::get_Q(double dT) {
    static double dTs[4];
    dTs[0] = dT;
    for (int i = 1; i < 4; ++i) {
        dTs[i] = dTs[i - 1] * dT;
    }
    double q_x_x = dTs[3] / 4 * Q2_XY, q_x_vx = dTs[2] / 2 * Q2_XY, q_vx_vx = dTs[1] * Q2_XY;
    double q_y_y = dTs[3] / 4 * Q2_YAW, q_y_vy = dTs[2] / 2 * Q2_YAW, q_vy_vy = dTs[1] * Q2_YAW;
    Q = Mnn::Zero();
    Q.block(0, 0, 2, 2) << q_x_x, q_x_vx,  //
        q_x_vx, q_vx_vx;                   //
    Q.block(2, 2, 2, 2) << q_x_x, q_x_vx,  //
        q_x_vx, q_vx_vx;                   //
    Q.block(4, 4, 2, 2) << q_y_y, q_y_vy,  //
        q_y_vy, q_vy_vy;                   //
}

void enemy_KF_4::get_R(const Output &output) {
    Vm R_vec;
    R_vec << abs(R_XYZ * output.x), abs(R_XYZ * output.y), abs(R_XYZ * output.z), R_YAW;
    R = R_vec.asDiagonal();
}

void enemy_KF_4::get_R(const Output2 &output) {
    Vm2 R_vec;
    R_vec << abs(R_XYZ * output.x), abs(R_XYZ * output.y), abs(R_XYZ * output.z), R_YAW, abs(R_XYZ * output.x2), abs(R_XYZ * output.y2),
        abs(R_XYZ * output.z2), R_YAW;
    R2 = R_vec.asDiagonal();
}

void enemy_KF_4::CKF_predict(double dT) {
    get_Q(dT);
    SRCR_sampling_3(Xe, Pe);
    Xp = Vn::Zero();
    for (int i = 0; i < sample_num; ++i) {
        sample_X[i] = f(samples[i], dT);
        Xp += weights[i] * sample_X[i];
    }

    Pp = Mnn::Zero();
    for (int i = 0; i < sample_num; ++i) {
        Pp += weights[i] * (sample_X[i] - Xp) * (sample_X[i] - Xp).transpose();
    }
    Pp += Q;
}

void enemy_KF_4::CKF_measure(const Vm &z, int phase_id) {
    sample_Z = std::vector<Vm>(sample_num);  // 修正
    Zp = Vm::Zero();
    for (int i = 0; i < sample_num; ++i) {
        sample_Z[i] = h(samples[i], phase_id);
        Zp += weights[i] * sample_Z[i];
    }

    Pzz = Mmm::Zero();
    for (int i = 0; i < sample_num; ++i) {
        Pzz += weights[i] * (sample_Z[i] - Zp) * (sample_Z[i] - Zp).transpose();
    }

    // 根据dis计算自适应R
    get_R(get_output(z));
    Pzz += R;
}

void enemy_KF_4::CKF_measure(const Vm2 &z, int phase_id, int phase_id2) {
    sample_Z2 = std::vector<Vm2>(sample_num);  // 修正
    Zp2 = Vm2::Zero();
    for (int i = 0; i < sample_num; ++i) {
        sample_Z2[i] = h(samples[i], phase_id, phase_id2);
        Zp2 += weights[i] * sample_Z2[i];
    }

    Pzz2 = Mmm2::Zero();
    for (int i = 0; i < sample_num; ++i) {
        Pzz2 += weights[i] * (sample_Z2[i] - Zp2) * (sample_Z2[i] - Zp2).transpose();
    }

    // 根据dis计算自适应R
    get_R(get_output(z));
    Pzz2 += R2;
}

void enemy_KF_4::CKF_correct(const Vm &z) {
    Pxz = Mnm::Zero();
    for (int i = 0; i < sample_num; ++i) {
        Pxz += weights[i] * (sample_X[i] - Xp) * (sample_Z[i] - Zp).transpose();
    }
    K = Pxz * Pzz.inverse();

    Xe = Xp + K * (z - Zp);
    Pe = Pp - K * Pzz * K.transpose();

    state = get_state(Xe);
}

void enemy_KF_4::CKF_correct(const Vm2 &z) {
    Pxz2 = Mnm2::Zero();
    for (int i = 0; i < sample_num; ++i) {
        Pxz2 += weights[i] * (sample_X[i] - Xp) * (sample_Z2[i] - Zp2).transpose();
    }
    K2 = Pxz2 * Pzz2.inverse();

    Xe = Xp + K2 * (z - Zp2);
    Pe = Pp - K * Pzz * K.transpose();

    state = get_state(Xe);
}

void enemy_KF_4::CKF_update(const Vm &z, double stamp, int phase_id) {
    double dT = stamp - timestamp;
    cout << "DT: " << dT << endl;
    Xe = get_X(state);
    PerfGuard perf_KF("KF");
    CKF_predict(dT);
    SRCR_sampling_3(Xp, Pp);
    CKF_measure(z, phase_id);
    CKF_correct(z);
    timestamp = stamp;
}

void enemy_KF_4::CKF_update(const Vm2 &z, double stamp, int phase_id, int phase_id2) {
    double dT = stamp - timestamp;
    Xe = get_X(state);
    PerfGuard perf_KF("KF");
    CKF_predict(dT);
    SRCR_sampling_3(Xp, Pp);
    CKF_measure(z, phase_id, phase_id2);
    CKF_correct(z);
    timestamp = stamp;
}
std::vector<Eigen::Vector3d> enemy_KF_4::predict_armors(double stamp) {
    State state_pre = predict(stamp);
    std::vector<Eigen::Vector3d> result;
    for (int i = 0; i < armor_cnt; ++i) {
        Output output_pre = get_output(h(get_X(state_pre), i));
        result.push_back(Eigen::Vector3d(output_pre.x, output_pre.y, output_pre.z));
    }
    return result;
}

void enemy_KF_4::load_params() {
    Pe = (Vn::Ones() * 0.1).asDiagonal();
    if (!is_declare_params) {
        R_XYZ = node->declare_parameter("R_XYZ", 0.01);
        R_YAW = node->declare_parameter("R_YAW", 0.01);
        Q2_XY = node->declare_parameter("Q2_XY", 0.01);
        Q2_YAW = node->declare_parameter("Q2_YAW", 0.01);
        is_declare_params = true;
    }
    cout << "R_XYZ" << R_XYZ << endl;
}