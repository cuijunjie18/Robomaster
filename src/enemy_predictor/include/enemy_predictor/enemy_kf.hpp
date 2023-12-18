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
class enemy_double_observer_EKF {
   public:
    // 带后缀2的名称用于表示同时观测两块装甲板时用到的类型或变量
    using Vn = Eigen::Matrix<double, 13, 1>;
    using Vm = Eigen::Matrix<double, 4, 1>;
    using Vm2 = Eigen::Matrix<double, 10, 1>;
    // using Vm_pts = Eigen::Matrix<double, 8, 1>;
    // using Vm_pts2 = Eigen::Matrix<double, 16, 1>;
    using Mnn = Eigen::Matrix<double, 13, 13>;
    using Mmm = Eigen::Matrix<double, 4, 4>;
    using Mmm2 = Eigen::Matrix<double, 10, 10>;
    // using Mmm_pts = Eigen::Matrix<double, 8, 8>;
    // using Mmm_pts2 = Eigen::Matrix<double, 16, 16>;
    using Mmn = Eigen::Matrix<double, 4, 13>;
    using Mmn2 = Eigen::Matrix<double, 10, 13>;
    // using Mmn_pts = Eigen::Matrix<double, 8, 13>;
    // using Mmn_pts2 = Eigen::Matrix<double, 16, 13>;
    using Mnm = Eigen::Matrix<double, 13, 4>;
    using Mnm2 = Eigen::Matrix<double, 13, 10>;
    // using Mnm_pts = Eigen::Matrix<double, 13, 8>;
    // using Mnm_pts2 = Eigen::Matrix<double, 13, 16>;

    struct config {
        Vn P;
        double R_XYZ, R_YAW;
        double Q2_XYZ, Q2_YAW, Q2_R;
    };
    struct Observe {
        double x;
        double y;
        double z;
        double yaw;
        Observe() {}
        Observe(double x_, double y_, double z_, double yaw_) : x(x_), y(y_), z(z_), yaw(yaw_) {}
    };
    struct Observe2 {
        double x;
        double y;
        double z;
        double r;
        double yaw;
        double x2;
        double y2;
        double z2;
        double r2;
        double yaw2;
        Observe2() {}
        Observe2(double x_, double y_, double z_, double r_, double yaw_, double x2_, double y2_, double z2_, double r2_, double yaw2_) : x(x_), y(y_), z(z_), r(r_), yaw(yaw_), x2(x2_), y2(y2_), z2(z2_), r2(r2_), yaw2(yaw2_) {}
    };
    // struct Observe_pts {
    //     double x1;
    //     double y1;
    //     double x2;
    //     double y2;
    //     double x3;
    //     double y3;
    //     double x4;
    //     double y4;
    //     Observe_pts() {}
    //     Observe_pts(double x1_, double y1_, double x2_, double y2_, double x3_, double y3_, double x4_, double y4_) : x1(x1_), y1(y1_), x2(x2_), y2(y2_), x3(x3_), y3(y3_), x4(x4_), y4(y4_) {}
    // };
    // struct Observe_pts2 {
    //     double x1a;
    //     double x1b;
    //     double y1a;
    //     double y1b;
    //     double x2a;
    //     double x2b;
    //     double y2a;
    //     double y2b;
    //     double x3a;
    //     double x3b;
    //     double y3a;
    //     double y3b;
    //     double x4a;
    //     double x4b;
    //     double y4a;
    //     double y4b;
    //     Observe_pts2() {}
    //     Observe_pts2(double x1_, double y1_, double x2_, double y2_, double x3_, double y3_, double x4_, double y4_, double x1__, double y1__, double x2__, double y2__, double x3__, double y3__, double x4__, double y4__) : x1a(x1_), y1a(y1_), x2a(x2_), y2a(y2_), x3a(x3_), y3a(y3_), x4a(x4_), y4a(y4_), x1b(x1__), y1b(y1__), x2b(x2__), y2b(y2__), x3b(x3__), y3b(y3__), x4b(x4__), y4b(y4__){}
    // };
    struct State {
        double x;
        double vx;
        double y;
        double vy;
        double z;
        double vz;
        double z2;
        double yaw;
        double vyaw;
        double yaw2;
        double vyaw2;
        double r;
        double r2;

        State(double x_, double vx_, double y_, double vy_, double z_, double vz_, double z2_, double yaw_, double vyaw_, double yaw2_, double vyaw2_, double r_, double r2_)
            : x(x_), vx(vx_), y(y_), vy(vy_), z(z_),  vz(vz_), z2(z2_), yaw(yaw_), vyaw(vyaw_), yaw2(yaw2_), vyaw2(vyaw2_), r(r_), r2(r2_) {}
        State(){};
    };

    static inline Vn get_X(State _state) {
        return Vn(_state.x, _state.vx, _state.y, _state.vy, _state.z, _state.vz, _state.z2, _state.yaw, _state.vyaw, _state.yaw2, _state.vyaw2, _state.r, _state.r2);
    }
    static inline State get_state(Vn _X) {
        return State(_X[0], _X[1], _X[2], _X[3], _X[4], _X[5], _X[6], _X[7], _X[8], _X[9], _X[10], _X[11], _X[12]);
    }
    static inline Vm get_Z(Observe _observe) {
        return Vm(_observe.x, _observe.y, _observe.z, _observe.yaw);
    }
    static inline Observe get_observe(Vm _Z) {
        return Observe(_Z[0], _Z[1], _Z[2], _Z[3]);
    }
    static inline Vm2 get_Z(Observe2 _observe) {
        return Vm2(_observe.x,_observe.y, _observe.z, _observe.r, _observe.yaw, _observe.x2,_observe.y2, _observe.z2, _observe.r2, _observe.yaw2);
    }
    static inline Observe2 get_observe(Vm2 _Z) {
        return Observe2(_Z[0], _Z[1], _Z[2], _Z[3], _Z[4], _Z[5], _Z[6], _Z[7], _Z[8], _Z[9]);
    }
    // static inline Vm_pts get_Z(Observe_pts _observe) {
    //     return Vm_pts(_observe.x1, _observe.y1, _observe.x2, _observe.y2, _observe.x3, _observe.y3, _observe.x4, _observe.y4);
    // }
    // static inline Observe_pts get_observe(Vm_pts _Z) {
    //     return Observe_pts(_Z[0], _Z[1], _Z[2], _Z[3], _Z[4], _Z[5], _Z[6], _Z[7]);
    // }
    // static inline Vm_pts2 get_Z(Observe_pts2 _observe) {
    //     return Vm_pts2(_observe.x1a, _observe.y1a, _observe.x2a, _observe.y2a, _observe.x3a, _observe.y3a, _observe.x4a, _observe.y4a, _observe.x1b, _observe.y1b, _observe.x2b, _observe.y2b, _observe.x3b, _observe.y3b, _observe.x4b, _observe.y4b);
    // }
    // static inline Observe_pts2 get_observe(Vm_pts2 _Z) {
    //     return Observe_pts2(_Z[0], _Z[1], _Z[2], _Z[3], _Z[4], _Z[5], _Z[6], _Z[7], _Z[8], _Z[9], _Z[10], _Z[11], _Z[12], _Z[13], _Z[14], _Z[15]);
    // }


    inline static Vn init_P;
    static constexpr int n = 13;    // 状态个数
    static constexpr int m = 4;    // 观测个数
    static constexpr int m2 = 10;  // 观测个数
    int sample_num;                // 不同采样模式有不同的样本数
    std::vector<Vn> samples;       // 样本数组
    std::vector<double> weights;   // 权重数组
    std::shared_ptr<Position_Calculator> pc_ptr;

    double last_r;
    int now_state_phase;
    rclcpp::Logger logger;
    State state;
    Vn Xe;  // 状态量
    // 0、1  X方向的位置、速度
    // 2、3  Y方向的位置、速度
    // 4、5、6 主装甲板高度，副装甲板高度、z方向速度
    // 7、8、9、10 两块装甲板的相位、角速度
    // 11、12 两块装甲板的半径
    std::vector<Vn> sample_X;  // 预测
    Vn Xp;
    Mnn Pp;
    Mnn Pe;
    Mnn Q;

    // std::vector<Vm_pts> sample_Z_pts;
    // Vm_pts Zp_pts;
    // Mmm_pts Pzz_pts;
    // Mnm_pts Pxz_pts;
    // Mmm_pts R_pts;
    // Mnm_pts K_pts;

    // std::vector<Vm_pts2> sample_Z_pts2;
    // Vm_pts2 Zp_pts2;
    // Mmm_pts2 Pzz_pts2;
    // Mnm_pts2 Pxz_pts2;
    // Mmm_pts2 R_pts2;
    // Mnm_pts2 K_pts2;

    std::vector<Vm> sample_Z;
    Vm Zp;
    Mmm Pzz;
    Mnm Pxz;
    Mmm R;
    Mnm K;

    std::vector<Vm2> sample_Z2;
    Vm2 Zp2;
    Mmm2 Pzz2;
    Mnm2 Pxz2;
    Mmm2 R2;
    Mnm2 K2;

    // 自适应参数
    inline static double R_XYZ, R_YAW;
    inline static double Q2_XYZ, Q2_YAW, Q2_R;


    // explicit enemy_double_observer_EKF(Position_Calculator *pc) : logger(rclcpp::get_logger("enemy_EKF")) {
    //     init();
    //     pc_ptr.reset(pc);
    // }

    explicit enemy_double_observer_EKF() : logger(rclcpp::get_logger("enemy_EKF")) {
        init();
    }

    void init() {
        sample_num = 2 * n;
        samples = std::vector<Vn>(sample_num);
        weights = std::vector<double>(sample_num);
        Pe = init_P.asDiagonal();
        now_state_phase = 0;
        sample_X = std::vector<Vn>(sample_num);
    }

    void reset(
        const Observe &observe,
        const bool& isMain = true,
        const double& _r1 = 0.2,
        const double& _r2 = 0.15) {
        Pe = init_P.asDiagonal();
        state.x = observe.x - _r1 * cos(observe.yaw),  //
        state.vx = 0,  //
        state.y = observe.y - _r1 * sin(observe.yaw),  //
        state.vy = 0,  //
        state.z = observe.z,  //
        state.vz = 0,  //
        state.z2 = observe.z,  //
        state.vyaw = 0,  //
        state.vyaw2 = 0,  //
        state.r = _r1,  //
        state.r2 = _r2;  //
        if (isMain) {
            state.yaw = observe.yaw;
            state.yaw2 = observe.yaw + ((state.yaw2 > state.yaw) ? 1 : -1) * M_PI_2;
        } else {
            state.yaw2 = observe.yaw;
            state.yaw = observe.yaw + ((state.yaw > state.yaw2) ? 1 : -1) * M_PI_2;
        }
        Xe = get_X(state);
    }

    void reset2(const Observe2 &observe) {
        Pe = init_P.asDiagonal();
        state.x = (observe.x-observe.r*cos(observe.yaw) + observe.x2-observe.r2*cos(observe.yaw2))/2,  //
        state.vx = 0,  //
        state.y = (observe.y-observe.r*sin(observe.yaw) + observe.y2-observe.r2*sin(observe.yaw2))/2,  //
        state.vy = 0,  //
        state.z = observe.z, 
        state.vz = 0,  //
        state.z2 = observe.z2,  //
        state.yaw = observe.yaw,  //
        state.vyaw = 0,  //
        state.yaw2 = observe.yaw2,  //
        state.vyaw2 = 0,  //
        state.r = observe.r,  //
        state.r2 = observe.r2;  //
        Xe = get_X(state);
    }


    Vn f(const Vn &X, double dT) const {
        // x,vx,y,vy,z1,vz,z2,theta1,w1,theta2,w2,r1,r2
        State _state = get_state(X);
        _state.x += _state.vx * dT;
        _state.y += _state.vy * dT;
        _state.z += _state.vz * dT;
        _state.z2 += _state.vz * dT;
        _state.yaw += _state.vyaw * dT;
        _state.yaw2 += _state.vyaw2 * dT;
        return get_X(_state);
    }

    Vm h(const Vn &X, bool isMain) {
        State _state = get_state(X);
        Observe _observe;
        double r;
        if (isMain) {
            _observe.yaw = _state.yaw;
            _observe.z = _state.z;
            r = _state.r;
        }
        else {
            _observe.yaw = _state.yaw2;
            _observe.z = _state.z2;
            r = _state.r2;
        }
        _observe.x = _state.x + r * cos(_observe.yaw);
        _observe.y = _state.y + r * sin(_observe.yaw);
        return get_Z(_observe);
    }

    Vm2 h2(const Vn &X) {
        State _state = get_state(X);
        Observe2 _observe;
        _observe.yaw = _state.yaw;
        _observe.yaw2 = _state.yaw2;
        _observe.r = _state.r;
        _observe.r2 = _state.r2;
        _observe.z = _state.z;
        _observe.z2 = _state.z2;
        _observe.x = _state.x + _state.r * cos(_state.yaw);
        _observe.x2 = _state.x + _state.r2 * cos(_state.yaw2);
        _observe.y = _state.y + _state.r * sin(_state.yaw);
        _observe.y2 = _state.y + _state.r2 * sin(_state.yaw2);
        return get_Z(_observe);
    }

    // Vm_pts h(const Vn &X, bool isBigArmor, bool isMain) {
    //     State _state = get_state(X);
    //     Observe_pts _observe;
    //     Eigen::Vector3d xyz;
    //     std::vector<cv::Point2d> armor_img;
    //     if (isMain) {
    //         xyz << _state.x + _state.r * cos(_state.yaw), _state.y + _state.r * sin(_state.yaw), _state.z;
    //         armor_img = pc_ptr->generate_armor_img(isBigArmor, -15.0, _state.yaw / M_PI * 180, xyz);
    //     } else {
    //         xyz << _state.x + _state.r2 * cos(_state.yaw2), _state.y + _state.r2 * sin(_state.yaw2), _state.z2;
    //         armor_img = pc_ptr->generate_armor_img(isBigArmor, -15.0, _state.yaw2 / M_PI * 180, xyz);
    //     }
    //     _observe.x1 = armor_img[0].x;
    //     _observe.y1 = armor_img[0].y;
    //     _observe.x2 = armor_img[1].x;
    //     _observe.y2 = armor_img[1].y;
    //     _observe.x3 = armor_img[2].x;
    //     _observe.y3 = armor_img[2].y;
    //     _observe.x4 = armor_img[3].x;
    //     _observe.y4 = armor_img[3].y;
    //     return get_Z(_observe);
    // }

    // Vm_pts2 h2(const Vn &X, bool isBigArmor) {
    //     State _state = get_state(X);
    //     Observe_pts2 _observe;
    //     Eigen::Vector3d xyz;
    //     xyz << _state.x + _state.r * cos(_state.yaw), _state.y + _state.r * sin(_state.yaw), _state.z;
    //     std::vector<cv::Point2d> armor_img = pc_ptr->generate_armor_img(isBigArmor, -15.0, _state.yaw / M_PI * 180, xyz);
    //     _observe.x1a = armor_img[0].x;
    //     _observe.y1a = armor_img[0].y;
    //     _observe.x2a = armor_img[1].x;
    //     _observe.y2a = armor_img[1].y;
    //     _observe.x3a = armor_img[2].x;
    //     _observe.y3a = armor_img[2].y;
    //     _observe.x4a = armor_img[3].x;
    //     _observe.y4a = armor_img[3].y;
    //     xyz << _state.x + _state.r2 * cos(_state.yaw2), _state.y + _state.r2 * sin(_state.yaw2), _state.z2;
    //     armor_img = pc_ptr->generate_armor_img(isBigArmor, -15.0, _state.yaw2 / M_PI * 180, xyz);
    //     _observe.x1b = armor_img[0].x;
    //     _observe.y1b = armor_img[0].y;
    //     _observe.x2b = armor_img[1].x;
    //     _observe.y2b = armor_img[1].y;
    //     _observe.x3b = armor_img[2].x;
    //     _observe.y3b = armor_img[2].y;
    //     _observe.x4b = armor_img[3].x;
    //     _observe.y4b = armor_img[3].y;

    //     return get_Z(_observe);
    // }

    State predict(double dT) { return get_state(f(Xe, dT)); }

    void CKF_predict(double dT) {
        //根据dT计算自适应Q
        static double dTs[4];
        dTs[0] = dT;
        for (int i = 1; i < 4; ++i) dTs[i] = dTs[i - 1] * dT;
        double q_x_x = dTs[3] / 4 * Q2_XYZ, q_x_vx = dTs[2] / 2 * Q2_XYZ, q_vx_vx = dTs[1] * Q2_XYZ;
        double q_y_y = dTs[3] / 4 * Q2_YAW, q_y_vy = dTs[2] / 2 * Q2_YAW, q_vy_vy = dTs[1] * Q2_YAW;
        double q_r_r = dTs[3] / 4 * Q2_R, q_r_vr = dTs[2] / 2 * Q2_R, q_vr_vr = dTs[1] * Q2_R;
        Q << q_x_x,  q_x_vx,  0.,     0.,      0.,     0.,      0.,     0.,     0.,      0.,     0.,      0.,    0.,    //
             q_x_vx, q_vx_vx, 0.,     0.,      0.,     0.,      0.,     0.,     0.,      0.,     0.,      0.,    0.,    //
             0.,     0.,      q_x_x,  q_x_vx,  0.,     0.,      0.,     0.,     0.,      0.,     0.,      0.,    0.,    //
             0.,     0.,      q_x_vx, q_vx_vx, 0.,     0.,      0.,     0.,     0.,      0.,     0.,      0.,    0.,    //
             0.,     0.,      0.,     0.,      q_r_r,  q_r_vr,  0.,     0.,     0.,      0.,     0.,      0.,    0.,    //
             0.,     0.,      0.,     0.,      q_r_vr, q_vr_vr, q_r_vr, 0.,     0.,      0.,     0.,      0.,    0.,    //
             0.,     0.,      0.,     0.,      0.,     q_r_vr,  q_r_r,  0.,     0.,      0.,     0.,      0.,    0.,    //
             0.,     0.,      0.,     0.,      0.,     0.,      0.,     q_y_y,  q_y_vy,  0.,     0.,      0.,    0.,    //
             0.,     0.,      0.,     0.,      0.,     0.,      0.,     q_y_vy, q_vy_vy, 0.,     0.,      0.,    0.,    //
             0.,     0.,      0.,     0.,      0.,     0.,      0.,     0.,     0.,      q_y_y,  q_y_vy,  0.,    0.,    //
             0.,     0.,      0.,     0.,      0.,     0.,      0.,     0.,     0.,      q_y_vy, q_vy_vy, 0.,    0.,    //
             0.,     0.,      0.,     0.,      0.,     0.,      0.,     0.,     0.,      0.,     0.,      q_r_r, 0.,    //
             0.,     0.,      0.,     0.,      0.,     0.,      0.,     0.,     0.,      0.,     0.,      0.,    q_r_r; //

        SRCR_sampling(Xe, Pe);
        
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

    // void CKF_update(const Vm_pts &z1, const Vm_pts &z2, bool isBigArmor, double dT) {
    //     Xe = get_X(state);
    //     PerfGuard perf_KF("KF");
    //     // round 1
    //     CKF_predict(dT);
    //     SRCR_sampling(Xp, Pp);
    //     CKF_measure(z1, isBigArmor, true);
    //     CKF_correct(z1);
    //     // round 2
    //     // reset
    //     sample_X = samples;
    //     Xp = Xe;
    //     Pp = Pe;
    //     Xp = Vn::Zero();
    //     for (int i = 0; i < sample_num; ++i) {
    //         Xp += weights[i] * sample_X[i];
    //     }
        
    //     Pp = Mnn::Zero();
    //     for (int i = 0; i < sample_num; ++i) {
    //         Pp += weights[i] * (sample_X[i] - Xp) * (sample_X[i] - Xp).transpose();
    //     }
    //     Pp += Q;
    //     SRCR_sampling(Xp,Pp);
    //     CKF_measure(z2, isBigArmor, false);
    //     CKF_correct(z2);
    // }
    // void CKF_update(const Vm_pts &z, bool isBigArmor, bool isMain, double dT) {
    //     Xe = get_X(state);
    //     PerfGuard perf_KF("KF");
    //     CKF_predict(dT);
    //     SRCR_sampling(Xp, Pp);
    //     CKF_measure(z, isBigArmor, isMain);
    //     CKF_correct(z);
    // }
    // void CKF_update2(const Vm_pts2 &z, bool isBigArmor, double dT) {
    //     Xe = get_X(state);
    //     PerfGuard perf_KF("KF");
    //     CKF_predict(dT);
    //     SRCR_sampling(Xp, Pp);
    //     CKF_measure2(z, isBigArmor);
    //     CKF_correct2(z);
    // }
    void CKF_measure(const Vm &z, bool isMain) {
        sample_Z = std::vector<Vm>(sample_num);  // 修正
        Zp = Vm::Zero();
        for (int i = 0; i < sample_num; ++i) {
            sample_Z[i] = h(samples[i], isMain);
            Zp += weights[i] * sample_Z[i];
        }

        Pzz = Mmm::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pzz += weights[i] * (sample_Z[i] - Zp) * (sample_Z[i] - Zp).transpose();
        }

        // 根据dis计算自适应R
        Observe _observe = get_observe(z);
        Vm R_vec;
        R_vec << abs(R_XYZ * _observe.x), abs(R_XYZ * _observe.y), abs(R_XYZ * _observe.z), abs(R_YAW * _observe.yaw);
        R = R_vec.asDiagonal();
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
    void CKF_measure2(const Vm2 &z) {
        sample_Z2 = std::vector<Vm2>(sample_num);  // 修正
        Zp2 = Vm2::Zero();
        for (int i = 0; i < sample_num; ++i) {
            sample_Z2[i] = h2(samples[i]);
            Zp2 += weights[i] * sample_Z2[i];
        }

        Pzz2 = Mmm2::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pzz2 += weights[i] * (sample_Z2[i] - Zp2) * (sample_Z2[i] - Zp2).transpose();
        }

        // 根据dis计算自适应R
        Observe2 _observe = get_observe(z);
        Vm2 R_vec;
        R_vec << abs(R_XYZ * _observe.x), abs(R_XYZ * _observe.y), abs(R_XYZ * _observe.z), abs(R_XYZ * _observe.r), abs(R_YAW * _observe.yaw),  abs(R_XYZ * _observe.x2), abs(R_XYZ * _observe.y2), abs(R_XYZ * _observe.z2), abs(R_XYZ * _observe.r2), abs(R_YAW * _observe.yaw2);
        R2 = R_vec.asDiagonal();
        Pzz2 += R2;
    }
    void CKF_correct(const Vm2 &z) {
        Pxz2 = Mnm2::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pxz2 += weights[i] * (sample_X[i] - Xp) * (sample_Z2[i] - Zp2).transpose();
        }
        K2 = Pxz2 * Pzz2.inverse();

        Xe = Xp + K2 * (z - Zp2);
        Pe = Pp - K2 * Pzz2 * K2.transpose();

        state = get_state(Xe);
    }
    void CKF_update(const Vm &z, double dT, bool isMain) {
        Xe = get_X(state);
        PerfGuard perf_KF("KF");
        CKF_predict(dT);
        SRCR_sampling(Xp, Pp);
        CKF_measure(z, isMain);
        CKF_correct(z);
    }
    void CKF_update2(const Vm2 &z, double dT) {
        Xe = get_X(state);
        PerfGuard perf_KF("KF");
        CKF_predict(dT);
        SRCR_sampling(Xp, Pp);
        CKF_measure2(z);
        CKF_correct(z);
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