#ifndef _EMENY_EKF_H_
#define _EMENY_EKF_H_
#include <enemy_predictor/EnPredictor_utils.h>
#include <math.h>

#include <Eigen/Cholesky>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <random>
#include <rclcpp/rclcpp.hpp>
#include <string>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

class armor_EKF {
   public:
    using Mxx = Eigen::Matrix<double, 6, 6>;
    using Myx = Eigen::Matrix<double, 3, 6>;
    using Mxy = Eigen::Matrix<double, 6, 3>;
    using Myy = Eigen::Matrix<double, 3, 3>;
    using Vx = Eigen::Matrix<double, 6, 1>;
    using Vy = Eigen::Matrix<double, 3, 1>;

    struct config{
        Vx P, Q;
        Vy R, Ke;
        int length;
    };
    // Xp = f(last_Xe) + Q
    // Xe = h(Xp) + R

    inline static Vx const_P, const_Q;
    inline static Vy const_R, const_Ke;
    inline static int length;

    Vx Xe;  // Xe 估计状态变量(滤波后的状态)
    Vx Xp;  // Xp 预测状态变量
    Mxx F;  // F 预测雅克比
    Myx H;  // H 观测雅克比
    Mxx P;  // P 状态协方差
    Mxx Q;  // Q 预测过程协方差
    Myy R;  // R 观测过程协方差
    Mxy K;  // K 卡尔曼增益
    Vy Yp;  // Yp 预测观测量
    Filter Aver_D;
    explicit armor_EKF(const Vx &X0 = Vx::Zero())
        : Xe(X0), P(Mxx::Identity()), Q(Mxx::Identity()), R(Myy::Identity()) {}

    void reset(const Eigen::Matrix<double, 3, 1> &tmp) {
        Xe[0] = tmp[0];
        Xe[1] = tmp[1];
        Xe[2] = tmp[2];
        Xe[3] = Xe[4] = Xe[5] = 0;
        P = const_P.asDiagonal();
        Q = const_Q.asDiagonal();
        R = const_R.asDiagonal();
        Aver_D = Filter(length);
    }
    Vy predict(double dT) const {  // 该函数只是单纯地计算 dT 秒之后的坐标
        Vy tmp;
        tmp[0] = Xe[0] + Xe[3] * dT * const_Ke[0];
        tmp[1] = Xe[1] + Xe[4] * dT * const_Ke[1];
        tmp[2] = Xe[2];
        return tmp;
    }

    void update(const Vy &Y, const double dT) {
        Xp = Xe;
        Xp[0] += Xe[3] * dT;
        Xp[1] += Xe[4] * dT;
        Xp[2] += Xe[5] * dT;
        F = Mxx::Identity();
        F(0, 3) = F(1, 4) = F(2, 5) = dT;
        P = F * P * F.transpose() + Q;

        Yp[0] = Xp[0];
        Yp[1] = Xp[1];
        Yp[2] = Xp[2];
        H = Myx::Zero();
        H(0, 0) = H(1, 1) = H(2, 2) = 1;

        K = P * H.transpose() * (H * P * H.transpose() + R).inverse();
        Xe = Xp + K * (Y - Yp);
        Aver_D.update((Y - Yp)[1] * (Y - Yp)[1] * Xe[2] * Xe[2]);
        P = (Mxx::Identity() - K * H) * P;
    }

    inline static void init(const config& config_) {
        const_P = config_.P;
        const_Q = config_.Q;
        const_R = config_.R;
        const_Ke = config_.Ke;
        length = config_.length;
    }
};

class enemy_half_observer_EKF {
   public:
    using Vn = Eigen::Matrix<double, 9, 1>;
    using Vm = Eigen::Matrix<double, 4, 1>;
    using Mnn = Eigen::Matrix<double, 9, 9>;
    using Mmm = Eigen::Matrix<double, 4, 4>;
    using Mmn = Eigen::Matrix<double, 4, 9>;
    using Mnm = Eigen::Matrix<double, 9, 4>;

    struct config{
        Vn P;
        double R_XYZ,R_YAW;
        double Q2_XYZ,Q2_YAW,Q2_R;
    };

    inline static Vn init_P;
    static constexpr int n = 9;  // 状态个数
    static constexpr int m = 4;  // 观测个数

    double last_r;
    int now_state_phase;
    rclcpp::Logger logger;
    Vn Xe;  // 状态量
    // 0、1  X方向的位置、速度
    // 2、3  Y方向的位置、速度
    // 4、5  车体自转的相位、角速度
    // 6、7  装甲板的高度,速度
    // 8     装甲板的自转半径（需要限位）

    // 自适应参数
    inline static double R_XYZ, R_YAW;
    inline static double Q2_XYZ, Q2_YAW, Q2_R;

    Mnn Pe;
    Mnn Q;
    Mmm R;
    Mnm K;

    explicit enemy_half_observer_EKF(): logger(rclcpp::get_logger("enemy_EKF")) {
        Pe = init_P.asDiagonal();
        now_state_phase = 0;
    }

    void reset(const Vm &observe) {
        Xe << observe[0], 0, observe[1], 0, observe[3], 0, observe[2], 0, 0.2;
        Pe = init_P.asDiagonal();
        last_r = Xe[8];
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
        result[8] = X[8];
        return result;
    }

    Vm h(const Vn &X) {
        Vm result = Vm::Zero();
        result[0] = X[0] + X[8] * cos(X[4]);
        result[1] = X[2] + X[8] * sin(X[4]);
        result[2] = X[6];
        result[3] = X[4];
        return result;
    }

    Vn predict(double dT) { return f(Xe, dT); }

    void update(const Vm &z, double dT) {
        Mnn F = Mnn::Zero();
        F << 1, dT, 0, 0, 0, 0, 0, 0, 0,  //
            0, 1, 0, 0, 0, 0, 0, 0, 0,    //
            0, 0, 1, dT, 0, 0, 0, 0, 0,   //
            0, 0, 0, 1, 0, 0, 0, 0, 0,    //
            0, 0, 0, 0, 1, dT, 0, 0, 0,   //
            0, 0, 0, 0, 0, 1, 0, 0, 0,    //
            0, 0, 0, 0, 0, 0, 1, dT, 0,   //
            0, 0, 0, 0, 0, 0, 0, 1, 0,    //
            0, 0, 0, 0, 0, 0, 0, 0, 1;    //
        Mmn H = Mmn::Zero();
        double r = Xe[8];
        double yaw = Xe[4];

        H << 1, 0, 0, 0, -r * sin(yaw), 0, 0, 0, cos(yaw), 0, 0, 1, 0, r * cos(yaw), 0, 0, 0,
            sin(yaw), 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
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
        Q << q_x_x, q_x_vx, 0, 0, 0, 0, 0, 0, 0,   //
            q_x_vx, q_vx_vx, 0, 0, 0, 0, 0, 0, 0,  //
            0, 0, q_x_x, q_x_vx, 0, 0, 0, 0, 0,    //
            0, 0, q_x_vx, q_vx_vx, 0, 0, 0, 0, 0,  //
            0, 0, 0, 0, q_y_y, q_y_vy, 0, 0, 0,    //
            0, 0, 0, 0, q_y_vy, q_vy_vy, 0, 0, 0,  //
            0, 0, 0, 0, 0, 0, q_x_x, q_x_vx, 0,    //
            0, 0, 0, 0, 0, 0, q_x_vx, q_vx_vx, 0,  //
            0, 0, 0, 0, 0, 0, 0, 0, q_r;
        // for(int i = 0;i < 4;++i){
        //     Q(i * 2,i * 2) = dTs[2] * init_Q[i + 1] / 3 + dTs[0] * init_Q[i];
        //     Q(i * 2,i * 2 + 1) = dTs[1] * init_Q[i + 1] / 2;
        //     Q(i * 2 + 1,i * 2) = dTs[1] * init_Q[i + 1] / 2;
        //     Q(i * 2 + 1,i * 2 + 1) = dTs[0] * init_Q[i + 1];
        // }
        // Q(8,8) = init_Q[8] * dTs[0];

        // Q[2][2] =
        Pe = F * Pe * F.transpose() + Q;
        Vn X_pri = f(Xe, dT);
        Vm Zp = h(X_pri);
        K = Pe * H.transpose() * (H * Pe * H.transpose() + R).inverse();
        Xe = X_pri + K * (z - Zp);
        Pe = (Mnn::Identity() - K * H) * Pe;

        // 进行R限幅
        if (Xe[8] < 0.15) {
            Xe[8] = 0.15;
        } else if (Xe[8] > 0.3) {
            Xe[8] = 0.3;
        }
    }

    double get_rotate_spd() { return Xe[5]; }

    double get_move_spd() { return sqrt(Xe[1] * Xe[1] + Xe[3] * Xe[3]); }

    inline static void init(const config& _config) {
        R_XYZ = _config.R_XYZ;
        R_YAW = _config.R_YAW;
        Q2_XYZ = _config.Q2_XYZ;
        Q2_YAW = _config.Q2_YAW;
        Q2_R = _config.Q2_R;
        init_P = _config.P;
        
    }
};
#endif