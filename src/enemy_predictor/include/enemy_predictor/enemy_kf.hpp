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
#include <rm_utils/perf.hpp>
#include <string>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
class enemy_KF {
   public:
    // 带后缀2的名称用于表示同时观测两块装甲板时用到的类型或变量
    using Vn = Eigen::Matrix<double, 11, 1>;
    using Vm = Eigen::Matrix<double, 4, 1>;
    using Vm2 = Eigen::Matrix<double, 10, 1>;
    using Mnn = Eigen::Matrix<double, 11, 11>;
    using Mmm = Eigen::Matrix<double, 4, 4>;
    using Mmm2 = Eigen::Matrix<double, 10, 10>;
    using Mmn = Eigen::Matrix<double, 4, 11>;
    using Mmn2 = Eigen::Matrix<double, 10, 11>;
    using Mnm = Eigen::Matrix<double, 11, 4>;
    using Mnm2 = Eigen::Matrix<double, 11, 10>;

    struct config {
        Vn P;
        double R_XYZ, R_YAW;
        double Q2_XYZ, Q2_YAW, Q2_R;
    };

    struct State {
        double x;
        double vx;
        double y;
        double vy;
        double yaw;
        double vyaw;
        double z;
        double vz;
        double z2;
        double r;
        double r2;

        State(double x_, double vx_, double y_, double vy_, double yaw_, double vyaw_, double z_, double vz_, double z2_, double r_, double r2_)
            : x(x_), vx(vx_), y(y_), vy(vy_), yaw(yaw_), vyaw(vyaw_), z(z_), vz(vz_), z2(z_), r(r_), r2(r_) {}
        State(){};
    };

    inline static Vn init_P;
    static constexpr int n = 11;   // 状态个数
    static constexpr int m = 4;    // 观测个数
    static constexpr int m2 = 10;  // 观测个数
    int sample_num;                // 不同采样模式有不同的样本数
    std::vector<Vn> samples;       // 样本数组
    std::vector<double> weights;   // 权重数组

    double last_r;
    int now_state_phase;
    rclcpp::Logger logger;
    State state;
    Vn Xe;  // 状态量
    // 0、1  X方向的位置、速度
    // 2、3  Y方向的位置、速度
    // 4、5  车体自转的相位、角速度
    // 6、7  装甲板的高度,速度
    // 8     额外装甲板的高度
    // 9     装甲板的自转半径（需要限位）
    // 10    额外装甲板的自转半径（需要限位）

    // 自适应参数
    inline static double R_XYZ, R_YAW;
    inline static double Q2_XYZ, Q2_YAW, Q2_R;

    Mnn Pe;
    Mnn Q;
    Mmm R;
    Mnm K;

    explicit enemy_KF() : logger(rclcpp::get_logger("enemy_KF")) {
        sample_num = 2 * n;
        samples = std::vector<Vn>(sample_num);
        weights = std::vector<double>(sample_num);
        Pe = init_P.asDiagonal();
        now_state_phase = 0;
    }

    Vn get_X(State _state) {
        Vn _X;
        _X[0] = _state.x;
        _X[1] = _state.vx;
        _X[2] = _state.y;
        _X[3] = _state.vy;
        _X[4] = _state.yaw;
        _X[5] = _state.vyaw;
        _X[6] = _state.z;
        _X[7] = _state.vz;
        _X[8] = _state.z2;
        _X[9] = _state.r;
        _X[10] = _state.r2;
        return _X;
    }
    State get_state(Vn _X) {
        State _state;
        _state.x = _X[0];
        _state.vx = _X[1];
        _state.y = _X[2];
        _state.vy = _X[3];
        _state.yaw = _X[4];
        _state.vyaw = _X[5];
        _state.z = _X[6];
        _state.vz = _X[7];
        _state.z2 = _X[8];
        _state.r = _X[9];
        _state.r2 = _X[10];
        return _state;
    }

    void reset(const Vm &observe) {
        state = State(observe[0], 0, observe[1], 0, observe[3], 0, observe[2], 0, observe[2], 0.2, 0.2);
        Xe = get_X(state);
        Pe = init_P.asDiagonal();
    }

    Vn f(const Vn &X, double dT) const {
        Vn result = Vn::Zero();
        result[0] = X[0] + X[1] * dT;
        result[1] = X[1];
        result[2] = X[2] + X[3] * dT;
        result[3] = X[3];
        result[4] = X[4] + X[5] * dT;
        result[5] = X[5];
        result[6] = X[6] + X[7] * dT;
        result[7] = X[7];
        result[8] = X[8] + X[7] * dT;
        result[9] = X[9];
        result[10] = X[10];
        return result;
    }

    Vm h(const Vn &X) {
        Vm result = Vm::Zero();
        result[0] = X[0] + X[9] * cos(X[4]);
        result[1] = X[2] + X[9] * sin(X[4]);
        result[2] = X[6];
        result[3] = X[4];
        return result;
    }

    State predict(double dT) { return get_state(f(Xe, dT)); }

    void SRCR_sampling(Vn _x, Mnn _P)  // 3阶球面——径向采样法
    {
        double sqrtn = sqrt(n);
        double weight = 1.0 / (2 * n);
        Eigen::LLT<Eigen::MatrixXd> get_S(_P);
        Eigen::MatrixXd S = get_S.matrixL();
        for (int i = 0; i < n; ++i) {
            samples[i] = _x + sqrtn * S.col(i);

            weights[i] = weight;

            samples[i + n] = _x - sqrtn * S.col(i);
            weights[i + n] = weight;
        }
    }

    void CKF_update(const Vm &z, double dT) {
        Xe = get_X(state);
        PerfGuard perf_KF("KF");
        // 根据dis计算自适应R
        Vm R_vec;
        R_vec << abs(R_XYZ * z[0]), abs(R_XYZ * z[1]), abs(R_XYZ * z[2]), R_YAW;
        R = R_vec.asDiagonal();
        // 根据dT计算自适应Q
        // logger.info("ekf_dt: {} {} {}",dT,dT,dT);
        static double dTs[4];
        dTs[0] = dT;
        for (int i = 1; i < 4; ++i) dTs[i] = dTs[i - 1] * dT;
        double q_x_x = dTs[3] / 4 * Q2_XYZ, q_x_vx = dTs[2] / 2 * Q2_XYZ, q_vx_vx = dTs[1] * Q2_XYZ;
        double q_y_y = dTs[3] / 4 * Q2_YAW, q_y_vy = dTs[2] / 2 * Q2_YAW, q_vy_vy = dTs[1] * Q2_YAW;
        double q_r = dTs[3] / 4 * Q2_R;
        //    xc      v_xc    yc      v_yc    yaw      v_yaw    za     v_za   r
        Q << q_x_x, q_x_vx, 0, 0, 0, 0, 0, 0, 0, 0, 0,        //
            q_x_vx, q_vx_vx, 0, 0, 0, 0, 0, 0, 0, 0, 0,       //
            0, 0, q_x_x, q_x_vx, 0, 0, 0, 0, 0, 0, 0,         //
            0, 0, q_x_vx, q_vx_vx, 0, 0, 0, 0, 0, 0, 0,       //
            0, 0, 0, 0, q_y_y, q_y_vy, 0, 0, 0, 0, 0,         //
            0, 0, 0, 0, q_y_vy, q_vy_vy, 0, 0, 0, 0, 0,       //
            0, 0, 0, 0, 0, 0, q_x_x, q_x_vx, 0, 0, 0,         //
            0, 0, 0, 0, 0, 0, q_x_vx, q_vx_vx, q_x_vx, 0, 0,  //
            0, 0, 0, 0, 0, 0, 0, q_x_vx, q_x_x, 0, 0,         //
            0, 0, 0, 0, 0, 0, 0, 0, 0, q_r, 0,                //
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, q_r;                //

        SRCR_sampling(Xe, Pe);

        std::vector<Vn> sample_X = std::vector<Vn>(sample_num);  // 预测
        Vn Xp = Vn::Zero();
        for (int i = 0; i < sample_num; ++i) {
            sample_X[i] = f(samples[i], dT);
            Xp += weights[i] * sample_X[i];
        }
        Mnn Pp = Mnn::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pp += weights[i] * (sample_X[i] - Xp) * (sample_X[i] - Xp).transpose();
        }
        Pp += Q;

        SRCR_sampling(Xp, Pp);

        std::vector<Vm> sample_Z = std::vector<Vm>(sample_num);  // 修正
        Vm Zp = Vm::Zero();
        for (int i = 0; i < sample_num; ++i) {
            sample_Z[i] = h(samples[i]);
            Zp += weights[i] * sample_Z[i];
        }

        Mmm Pzz = Mmm::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pzz += weights[i] * (sample_Z[i] - Zp) * (sample_Z[i] - Zp).transpose();
        }
        Pzz += R;

        Mnm Pxz = Mnm::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pxz += weights[i] * (sample_X[i] - Xp) * (sample_Z[i] - Zp).transpose();
        }

        K = Pxz * Pzz.inverse();

        Xe = Xp + K * (z - Zp);
        Pe = Pp - K * Pzz * K.transpose();

        state = get_state(Xe);
        // 进行R限幅
        if (state.r < 0.15) {
            state.r = 0.15;
        } else if (state.r > 0.3) {
            state.r = 0.3;
        }
        Xe = get_X(state);
    }

    double get_rotate_spd() { return state.vyaw; }

    double get_move_spd() { return sqrt(state.vx * state.vx + state.vy * state.vy); }

    inline static void init(const config &_config) {
        R_XYZ = _config.R_XYZ;
        R_YAW = _config.R_YAW;
        Q2_XYZ = _config.Q2_XYZ;
        Q2_YAW = _config.Q2_YAW;
        Q2_R = _config.Q2_R;
        init_P = _config.P;
    }
};
#endif