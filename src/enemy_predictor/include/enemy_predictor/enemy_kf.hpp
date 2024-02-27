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

using std::cout;
using std::endl;

const int state_num = 6;
const int output_num = 4;  // z坐标的处理有点不标准，但是看起来不影响运行
const int output_num2 = 8;
class enemy_KF_4 {
   public:
    // n代表状态维数，m代表输出维数
    using Vn = Eigen::Vector<double, state_num>;
    using Vm = Eigen::Vector<double, output_num>;
    using Vm2 = Eigen::Vector<double, output_num2>;
    using Mnn = Eigen::Matrix<double, state_num, state_num>;
    using Mmm = Eigen::Matrix<double, output_num, output_num>;
    using Mmm2 = Eigen::Matrix<double, output_num2, output_num2>;
    using Mmn = Eigen::Matrix<double, output_num, state_num>;
    using Mmn2 = Eigen::Matrix<double, output_num2, state_num>;
    using Mnm = Eigen::Matrix<double, state_num, output_num>;
    using Mnm2 = Eigen::Matrix<double, state_num, output_num2>;
    // 输出量的顺序是x,y,z,Re,Im
    explicit enemy_KF_4(rclcpp::Node *node_);

    struct State {
        double x, vx, y, vy;
        double yaw, omega;
        State(){};
        State(double X, double VX, double Y, double VY, double YAW, double OMEGA) {
            x = X;
            y = Y;
            yaw = YAW;
            vx = VX;
            vy = VY;
            omega = OMEGA;
        }
    };

    struct Output {
        double x, y, z;
        double yaw;
        // int phase_id;
        Output() {}
        Output(double X, double Y, double Z, double YAW) {
            x = X;
            y = Y;
            z = Z;
            yaw = YAW;
            // phase_id = PHASE_ID;
        }
    };

    struct Output2 {
        double x, y, z;
        double yaw;
        double dis;
        double x2, y2, z2;
        double yaw2;
        // int phase_id;
        Output2() {}
        Output2(double X, double Y, double Z, double YAW, double X2, double Y2, double Z2, double YAW2) {
            x = X;
            y = Y;
            z = Z;
            yaw = YAW;
            x2 = X2;
            y2 = Y2;
            z2 = Z2;
            yaw2 = YAW2;
            // phase_id = PHASE_ID;
        }
    };

    void load_params();

    Vn get_X(State _state) const;

    State get_state(Vn _X) const;

    Vm get_Z(Output _output);

    Vm2 get_Z(Output2 _output);

    Output get_output(Vm _Z);

    Output2 get_output(Vm2 _Z);

    void reset(const Output &observe, int phase_id, int armor_cnt, double stamp);

    Vn f(const Vn &X, double dT) const;

    Vm h(const Vn &X, int phase_id);

    Vm2 h(const Vn &X, int phase_id, int phase_id2);

    State predict(double stamp) { return get_state(f(Xe, stamp - timestamp)); }

    void SRCR_sampling_3(Vn _x, Mnn _P);  // 3阶球面——径向采样法

    void get_Q(double dT);

    void get_R(const Output &output);

    void get_R(const Output2 &output);

    void CKF_predict(double dT);

    void CKF_measure(const Vm &z, int phase_id);

    void CKF_measure(const Vm2 &z, int phase_id, int phase_id2);

    void CKF_correct(const Vm &z);

    void CKF_correct(const Vm2 &z);

    void CKF_update(const Vm &z, double stamp, int phase_id);

    void CKF_update(const Vm2 &z, double stamp, int phase_id, int phase_id2);

    Eigen::Vector3d get_center(State state_) { return Eigen::Vector3d(state_.x, state_.y, 0); }
    Eigen::Vector3d get_armor(State state_, int phase_id) {
        Output now_output = get_output(h(get_X(state_), phase_id));
        return Eigen::Vector3d(now_output.x, now_output.y, now_output.z);
    }

    std::vector<Eigen::Vector3d> predict_armors(double stamp);

    int sample_num;
    std::vector<double> const_dis;
    std::vector<double> const_z;
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
    Mmm2 R2;
    std::vector<Vm2> sample_Z2;
    Vm2 Zp2;
    Mmm2 Pzz2;
    Mnm2 Pxz2;
    Mnm2 K2;
    double timestamp;
    double armor_cnt;
    double angle_dis;
    inline static Vn init_P;
    inline static double R_XYZ, R_YAW;
    inline static double Q2_XY, Q2_YAW;
    inline static bool is_declare_params = false;

    rclcpp::Node *node;
};

#endif