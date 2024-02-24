#ifndef _EMENY_KF_H_
#define _EMENY_KF_H_
#include <enemy_predictor/EnPredictor_utils.h>
#include <math.h>

#include <Eigen/Cholesky>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <random>
#include <rclcpp/rclcpp.hpp>
#include <rm_utils/Position_Calculator.hpp>
#include <rm_utils/perf.hpp>
#include <string>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

const int state_num = 15;
const int output_num = 5;
const int armor_num = 4;
const double angle_dis = M_PI * 2 / armor_num;
class enemy_KF_4 {
   public:
    // n代表状态维数，m代表输出维数
    using Vn = Eigen::Vector<double, state_num>;
    using Vm = Eigen::Vector<double, output_num>;
    using Mnn = Eigen::Matrix<double, state_num, state_num>;
    using Mmm = Eigen::Matrix<double, output_num, output_num>;
    using Mmn = Eigen::Matrix<double, output_num, state_num>;
    using Mnm = Eigen::Matrix<double, state_num, output_num>;
    // 输出量的顺序是x,y,z,Re,Im
    explicit enemy_KF_4() : logger(rclcpp::get_logger("enemy_EKF")) {
        sample_num = 2 * state_num;
        samples = std::vector<Vn>(sample_num);
        weights = std::vector<double>(sample_num);
        Pe = init_P.asDiagonal();

        R_XYZ = 0.01, R_ReIm = 0.005, Q2_XY = 0.01, Q2_Z = 0.01, Q2_OMEGA = 0.01, Q2_DIS = 0.001, Q2_ReIm = 0.001;
    }

    struct State {
        double x, vx, y, vy;
        double Re, Im;
        double omega;
        std::vector<double> dis;
        std::vector<double> z;
        State(){};
        State(double X, double VX, double Y, double VY, double RE, double IM, double OMEGA, std::vector<double> DIS, std::vector<double> Z) {
            x = X;
            y = Y;
            Re = RE;
            Im = IM;
            vx = VX;
            vy = VY;
            omega = OMEGA;
            dis = DIS;
            z = Z;
        }
    };

    struct Output {
        double x, y, z;
        double Re, Im;
        // int phase_id;
        Output() {}
        Output(double X, double Y, double Z, double RE, double IM) {
            x = X;
            y = Y;
            z = Z;
            Re = RE;
            Im = IM;
            // phase_id = PHASE_ID;
        }
    };

    Vn get_X(State _state) const {
        Vn _X;
        _X[0] = _state.x;
        _X[1] = _state.vx;
        _X[2] = _state.y;
        _X[3] = _state.vy;
        _X[4] = _state.Re;
        _X[5] = _state.Im;
        _X[6] = _state.omega;

        int index = 7;
        for (double d : _state.dis) {
            _X[index++] = d;
        }

        for (double z_value : _state.z) {
            _X[index++] = z_value;
        }

        return _X;
    }

    State get_state(Vn _X) const {
        State _state;
        _state.x = _X[0];
        _state.vx = _X[1];
        _state.y = _X[2];
        _state.vy = _X[3];
        _state.Re = _X[4];
        _state.Im = _X[5];
        _state.omega = _X[6];

        _state.dis.resize(armor_num);
        for (int i = 0; i < armor_num; ++i) {
            _state.dis[i] = _X[7 + i];
        }

        _state.z.resize(armor_num);
        for (int i = 0; i < armor_num; ++i) {
            _state.z[i] = _X[7 + armor_num + i];
        }

        return _state;
    }

    Vm get_Z(Output _output) {
        Vm result;
        result[0] = _output.x;
        result[1] = _output.y;
        result[2] = _output.z;
        result[3] = _output.Re;
        result[4] = _output.Im;
        return result;
    }

    Output get_output(Vm _Z) {
        Output result;
        result.x = _Z[0];
        result.y = _Z[1];
        result.z = _Z[2];
        result.Re = _Z[3];
        result.Im = _Z[4];
        return result;
    }

    void reset(const Output &observe, int phase_id) {
        std::vector<double> dis(4, 0.2);
        std::vector<double> z(4, 0.1);
        state = State(observe.x, 0, observe.y, 0, 0, 0, 0, dis, z);
        state.Re = cos(atan2(observe.Im, observe.Re) - phase_id * angle_dis);
        state.Im = sin(atan2(observe.Im, observe.Re) - phase_id * angle_dis);
        Xe = get_X(state);
        Pe = init_P.asDiagonal();
    }

    Vn f(const Vn &X, double dT) const {
        State X_state = get_state(X);
        X_state.x = X_state.x + X_state.vx * dT;
        X_state.y = X_state.y + X_state.vy * dT;
        X_state.Re = cos(atan2(X_state.Im, X_state.Re) + X_state.omega * dT);
        X_state.Im = sin(atan2(X_state.Im, X_state.Re) + X_state.omega * dT);
        Vn result = get_X(X_state);
        return result;
    }
    Vm h(const Vn &X, int phase_id) {
        State X_state = get_state(X);
        Output Z_output;
        Z_output.Re = cos(atan2(X_state.Im, X_state.Re) + phase_id * angle_dis);
        Z_output.Im = sin(atan2(X_state.Im, X_state.Re) + phase_id * angle_dis);
        Z_output.x = X_state.x + X_state.dis[phase_id] * Z_output.Re;
        Z_output.y = X_state.y + X_state.dis[phase_id] * Z_output.Im;
        Z_output.z = X_state.z[phase_id];
        Vm result = get_Z(Z_output);
        return result;
    }

    State predict(double dT) { return get_state(f(Xe, dT)); }

    void SRCR_sampling_3(Vn _x, Mnn _P)  // 3阶球面——径向采样法
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

    void get_Q(double dT) {
        static double dTs[4];
        dTs[0] = dT;
        for (int i = 1; i < 4; ++i) {
            dTs[i] = dTs[i - 1] * dT;
        }
        double q_x_x = dTs[3] / 4 * Q2_XY, q_x_vx = dTs[2] / 2 * Q2_XY, q_vx_vx = dTs[1] * Q2_XY;
        Q = Mnn::Zero();
        Q.block(0, 0, 2, 2) << q_x_x, q_x_vx,  //
            q_x_vx, q_vx_vx;                   //
        Q.block(2, 2, 2, 2) << q_x_x, q_x_vx,  //
            q_x_vx, q_vx_vx;                   //
        Eigen::Vector3d Q_yaw_vec;
        Q_yaw_vec << Q2_ReIm, Q2_ReIm, Q2_OMEGA;
        Q.block(4, 4, 3, 3) = Q_yaw_vec.asDiagonal();
        Eigen::Vector4d Q_dis_vec;
        Q_dis_vec << Q2_DIS, Q2_DIS, Q2_DIS, Q2_DIS;
        Q.block(7, 7, 4, 4) = Q_dis_vec.asDiagonal();
        Eigen::Vector4d Q_z_vec;
        Q_z_vec << Q2_Z, Q2_Z, Q2_Z, Q2_Z;
        Q.block(11, 11, 4, 4) = Q_z_vec.asDiagonal();
    }

    void get_R(const Output &output) {
        Vm R_vec;
        R_vec << abs(R_XYZ * output.x), abs(R_XYZ * output.y), abs(R_XYZ * output.z), R_ReIm, R_ReIm;
        R = R_vec.asDiagonal();
    }

    void CKF_predict(double dT) {
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

    void CKF_measure(const Vm &z, int phase_id) {
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
    void CKF_correct(const Vm &z) {
        Pxz = Mnm::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pxz += weights[i] * (sample_X[i] - Xp) * (sample_Z[i] - Zp).transpose();
        }
        K = Pxz * Pzz.inverse();

        Xe = Xp + K * (z - Zp);
        Pe = Pp - K * Pzz * K.transpose();

        state = get_state(Xe);
    }
    void CKF_update(const Vm &z, double dT, int phase_id) {
        Xe = get_X(state);
        PerfGuard perf_KF("KF");
        CKF_predict(dT);
        SRCR_sampling_3(Xp, Pp);
        CKF_measure(z, phase_id);
        CKF_correct(z);
    }

    Eigen::Vector3d get_center(State state_) { return Eigen::Vector3d(state_.x, state_.y, 0); }
    Eigen::Vector3d get_armor(State state_, int phase_id) {
        Output now_output = get_output(h(get_X(state_), phase_id));
        return Eigen::Vector3d(now_output.x, now_output.y, now_output.z);
    }

    int sample_num;
    std::vector<Vn> samples;      // 样本数组
    std::vector<double> weights;  // 权重数组
    rclcpp::Logger logger;
    State state;
    Vn Xe;  // 状态量
    // 自适应参数
    Vn Xp;
    Mnn Pp;
    std::vector<Vn> sample_X;  // 预测
    Mnn Pe;
    Mnn Q;
    Mmm R;
    std::vector<Vm> sample_Z;
    Vm Zp;
    Mmm Pzz;
    Mnm Pxz;
    Mnm K;
    inline static Vn init_P;
    inline static double R_XYZ, R_ReIm;
    inline static double Q2_XY, Q2_OMEGA, Q2_DIS, Q2_Z, Q2_ReIm;
};

#endif