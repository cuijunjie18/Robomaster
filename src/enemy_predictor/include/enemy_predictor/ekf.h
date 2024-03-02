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
#include <rm_utils/perf.hpp>
#include <string>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
using std::cout;
using std::endl;
class armor_EKF {
   public:
    using Mxx = Eigen::Matrix<double, 6, 6>;
    using Myx = Eigen::Matrix<double, 3, 6>;
    using Mxy = Eigen::Matrix<double, 6, 3>;
    using Myy = Eigen::Matrix<double, 3, 3>;
    using Vx = Eigen::Matrix<double, 6, 1>;
    using Vy = Eigen::Matrix<double, 3, 1>;

    struct config {
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
    explicit armor_EKF(const Vx &X0 = Vx::Zero()) : Xe(X0), P(Mxx::Identity()), Q(Mxx::Identity()), R(Myy::Identity()) {}

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
        tmp[2] = Xe[2] + Xe[4] * dT * const_Ke[2];
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

    inline static void init(const config &config_) {
        const_P = config_.P;
        const_Q = config_.Q;
        const_R = config_.R;
        const_Ke = config_.Ke;
        length = config_.length;
    }
};

class yaw_KF {
   public:
    using Mxx = Eigen::Matrix<double, 2, 2>;
    using Myx = Eigen::Matrix<double, 1, 2>;
    using Mxy = Eigen::Matrix<double, 2, 1>;
    using Myy = Eigen::Matrix<double, 1, 1>;
    using Vx = Eigen::Matrix<double, 2, 1>;
    using Vy = Eigen::Matrix<double, 1, 1>;

    struct config {
        Vx P, Q;
        Vy R;
        double sigma2_Q;
    };

    inline static Vx const_P, const_Q;
    inline static Vy const_R, const_Ke;
    inline static int length;
    inline static double sigma2_Q;

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
    explicit yaw_KF(const Vx &X0 = Vx::Zero()) : Xe(X0), P(Mxx::Identity()), Q(Mxx::Identity()), R(Myy::Identity()) {
        // const_P = Eigen::Vector2d(1.0, 1.0);
        // sigma2_Q = 1;
        // const_R << 0.0029;
    }

    void reset(const Eigen::Matrix<double, 1, 1> &tmp) {
        Xe[0] = tmp[0];
        Xe[1] = 0;
        P = const_P.asDiagonal();
        R = const_R.asDiagonal();
    }
    Vy predict(double dT) const {  // 该函数只是单纯地计算 dT 秒之后的坐标
        Vy tmp;
        tmp[0] = Xe[0] + Xe[1] * dT;
        return tmp;
    }

    void update(const Vy &Y, const double dT) {
        static double dTs[4];
        dTs[0] = dT;
        for (int i = 1; i < 4; ++i) dTs[i] = dTs[i - 1] * dT;
        double q_x_x = dTs[3] / 4 * sigma2_Q, q_x_vx = dTs[2] / 2 * sigma2_Q, q_vx_vx = dTs[1] * sigma2_Q;
        Q << q_x_x, q_x_vx, q_x_vx, q_vx_vx;

        cout << "params" << sigma2_Q << endl << endl << Q << endl << endl << R << endl << endl;

        Xp = Xe;
        Xp[0] += Xe[1] * dT;
        F = Mxx::Identity();
        F(0, 1) = dT;
        P = F * P * F.transpose() + Q;

        Yp[0] = Xp[0];
        H = Myx::Zero();
        H(0, 0) = 1;

        K = P * H.transpose() * (H * P * H.transpose() + R).inverse();
        Xe = Xp + K * (Y - Yp);
        P = (Mxx::Identity() - K * H) * P;
    }

    inline static void init(const config &config_) {
        const_P = config_.P;
        sigma2_Q = config_.sigma2_Q;
        const_R = config_.R;
    }
};
#endif