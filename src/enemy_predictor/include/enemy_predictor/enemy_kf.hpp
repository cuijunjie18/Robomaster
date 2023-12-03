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

/*
class enemy_double_observer_EKF
{

    // 整车建模
    // x->x_1, y->y_1 ? 角速度与平动速度会耦合到v里
    // X  = [x, v_x, a_x, y, v_y, a_y, z_1, z_2, v_z, theta_1, v_yaw1, a_yaw1, theta_2, v_yaw2, a_yaw2, r_1, r_2]
    // Z2 = [x_1, y_1, z_1, r_1, theta_1, x_2, y_2, z_2, r_2, theta_2]
    // Z  = [x_1, y_1, z_1, theta_1]
    // 观测与状态转移独立，两个观测方程不影响predict

    public:
        using Mnn      =     Eigen::Matrix<double, 17, 17>;
        using Mmm2     =     Eigen::Matrix<double, 10, 10>;
        using Mnm2     =     Eigen::Matrix<double, 17, 10>;
        using Mmn2     =     Eigen::Matrix<double, 10, 17>;
        using Vn       =     Eigen::Matrix<double, 17, 1>;
        using Vm2      =     Eigen::Matrix<double, 10, 1>;
        using Mmm      =     Eigen::Matrix<double, 4, 4>;
        using Mnm      =     Eigen::Matrix<double, 17, 4>;
        using Mmn      =     Eigen::Matrix<double, 4, 17>;
        using Vm       =     Eigen::Matrix<double, 4, 1>;

        struct State {
            double x;
            double vx;
            double ax;
            double y;
            double vy;
            double ay;
            double yaw;
            double vyaw;
            double ayaw;
            double yaw2;
            double vyaw2;
            double ayaw2;
            double z;
            double vz;
            double z2;
            double r;
            double r2;
            State(double x_, double vx_, double ax_, double y_, double vy_, double ay_, double z_, double z2_, double vz_, double yaw_, double vyaw_, double ayaw_, double yaw2_, double vyaw2_, double ayaw2_, double r_, double r2_) : x(x_), vx(vx_), ax(ax_), y(y_), vy(vy_), ay(ay_), yaw(yaw_), vyaw(vyaw_), ayaw(ayaw_), yaw2(yaw2_), vyaw2(vyaw2_), ayaw2(ayaw2_), z(z_), vz(vz_), z2(z2_), r(r_), r2(r2_) {} 
            State(){}
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
        struct config {
            Vn P;
            double R_XYZ, R_YAW;
            double Q2_XYZ, Q2_YAW, Q2_R;
        };

        static inline Vn get_X(State _state) {
            return Vn(_state.x, _state.vx, _state.ax, _state.y, _state.vy, _state.ay, _state.z, _state.z2, _state.vz, _state.yaw, _state.vyaw, _state.ayaw, _state.yaw2, _state.vyaw2, _state.ayaw2, _state.r, _state.r2);
        }
        static inline State get_state(Vn _X) {
            return State(_X[0],_X[1],_X[2],_X[3],_X[4],_X[5],_X[6],_X[7],_X[8],_X[9],_X[10],_X[11],_X[12],_X[13],_X[14],_X[15],_X[16]);
        }

        explicit enemy_double_observer_EKF(
            const Vn& X0 = Vn::Zero(),
            const double& pval = 0.1,
            const double& qval = 1e-4,
            const double& rval = 1e-5)
            : X_prior(X0),
              X_posterior(X0),
              P_prior(Mnn::Identity() * pval),
              P_posterior(Mnn::Identity() * pval),
              Q_prior(Mnn::Identity() * qval),
              Q_posterior(Mnn::Identity() * qval),
              R_prior(Mmm::Identity() * rval),
              R_posterior(Mmm::Identity() * qval),
              R2_prior(Mmm2::Identity() * rval),
              R2_posterior(Mmm2::Identity() * qval),
              alpha(1.),
              b(0.9),
              state(get_state(X0)),
              logger(rclcpp::get_logger("enemy_KF")) {}

        inline static void init(const config &_config) {
            // TODO
        }

        void reset(
            const Observe &observe,
            const double& pval = 0.1,
            const double& qval = 1e-4,
            const double& rval = 1e-5,
            const double& _r1 = 0.2,
            const double& _r2 = 0.15) {
            P_prior = Mnn::Identity() * pval;
            P_posterior = Mnn::Identity() * pval;
            Q_prior = Mnn::Identity() * qval;
            R_prior = Mmm::Identity() * rval;
            Xe << observe.x - _r1 * cos(observe.yaw), //
                0,  //
                observe.y - _r1 * sin(observe.yaw), //
                0, //
                observe.z, //
                0, //
                observe.z, //
                observe.yaw, //
                0, //
                observe.yaw + ((angle_normalize(state.yaw) > angle_normalize(state.yaw2)) ? 1 : -1) * M_PI_2, // 
                0, //
                _r1,
                _r2;
            state = get_state(Xe);
        }
        void reset2(
            const Observe &observe,
            const double& pval = 0.1,
            const double& qval = 1e-4,
            const double& rval = 1e-5) {
            P_prior = Mnn::Identity() * pval;
            P_posterior = Mnn::Identity() * pval;
            Q_prior = Mnn::Identity() * qval;
            R2_prior = Mmm2::Identity() * rval;) {
            Xe << (observe.x-observe.r*cos(observe.yaw) + observe.x2-observe.r2*cos(observe.yaw2))/2,
                        0,
                        (observe.y-observe.r*sin(observe.yaw) + observe.y2-observe.r2*sin(observe.yaw2))/2,
                        0, observe.z, 0, observe.z2, observe.yaw, 0, observe.yaw2, 0, observe.r, observe.r2;
            state = get_state(Xe);
        }

        void log(char* msg, const Vn& _X) {
            RCLCPP_WARN(logger, "%s: %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf", msg, _X[0],_X[1],_X[2],_X[3],_X[4],_X[5],_X[6],_X[7],_X[8],_X[9],_X[10],_X[11],_X[12],_X[13],_X[14],_X[15],_X[16]);
        }
        void log(char* msg, const Vm& _Z) {
            RCLCPP_WARN(logger, "%s: %lf, %lf, %lf, %lf", msg, _Z[0],_Z[1],_Z[2],_Z[3]);
        }
        void log(char* msg, const Vm2& _Z2) {
            RCLCPP_WARN(logger, "%s: %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf", msg,
            _Z2[0],_Z2[1],_Z2[2],_Z2[3],_Z2[4],_Z2[5],_Z2[6],_Z2[7],_Z2[8],_Z2[9]);
        }

        void f(const double &dt) {
            double o_dt2 = 1.0 / 2 * dt * dt;
            F << 1, dt, o_dt2, 0,  0,  0,     0,  0,  0,  0, 0,  0,     0, 0,  0,     0, 0, //
                 0, 1,  dt,    0,  0,  0,     0,  0,  0,  0, 0,  0,     0, 0,  0,     0, 0, //
                 0, 0,  1,     0,  0,  0,     0,  0,  0,  0, 0,  0,     0, 0,  0,     0, 0, //
                 0, 0,  0,     1,  dt, o_dt2, 0,  0,  0,  0, 0,  0,     0, 0,  0,     0, 0, //
                 0, 0,  0,     0,  1,  dt,    0,  0,  0,  0, 0,  0,     0, 0,  0,     0, 0, //
                 0, 0,  0,     0,  0,  1,     0,  0,  0,  0, 0,  0,     0, 0,  0,     0, 0, //
                 0, 0,  0,     0,  0,  0,     1,  0,  dt, 0, 0,  0,     0, 0,  0,     0, 0, //
                 0, 0,  0,     0,  0,  0,     0,  1,  dt, 0, 0,  0,     0, 0,  0,     0, 0, //
                 0, 0,  0,     0,  0,  0,     0,  0,  1,  0, 0,  0,     0, 0,  0,     0, 0, //
                 0, 0,  0,     0,  0,  0,     0,  0,  0,  1, dt, o_dt2, 0, 0,  0,     0, 0, //
                 0, 0,  0,     0,  0,  0,     0,  0,  0,  0, 1,  dt,    0, 0,  0,     0, 0, //
                 0, 0,  0,     0,  0,  0,     0,  0,  0,  0, 0,  1,     0, 0,  0,     0, 0, //
                 0, 0,  0,     0,  0,  0,     0,  0,  0,  0, 0,  0,     1, dt, o_dt2, 0, 0, //
                 0, 0,  0,     0,  0,  0,     0,  0,  0,  0, 0,  0,     0, 1,  dt,    0, 0, //
                 0, 0,  0,     0,  0,  0,     0,  0,  0,  0, 0,  0,     0, 0,  1,     0, 0, //
                 0, 0,  0,     0,  0,  0,     0,  0,  0,  0, 0,  0,     0, 0,  0,     1, 0, //
                 0, 0,  0,     0,  0,  0,     0,  0,  0,  0, 0,  0,     0, 0,  0,     0, 1; //
            X_prior = F * X_posterior;
            log("f_X_prior", X_prior);
        }

        Vm2 h2(const Vn &_X) {
            State _state = get_state(_X)
            double r1 = _state.r;
            double r2 = _state.r2;
            double theta_1 = _state.yaw;
            double theta_2 = _state.yaw2;
            // X  = [x, v_x, a_x, y, v_y, a_y, z_1, z_2, v_z, theta_1, v_yaw1, a_yaw1, theta_2, v_yaw2, a_yaw2, r_1, r_2]
            // Z2 = [x_1, y_1, z_1, r_1, theta_1, x_2, y_2, z_2, r_2, theta_2]
            H2 << 1,  0,  0,  0,  0,  0,  0, 0, 0, -1*r1*sin(theta_1), 0, 0, 0,                  0, 0, cos(theta_1), 0,            //
                  0,  0,  0,  1,  0,  0,  0, 0, 0, r1*cos(theta_1),    0, 0, 0,                  0, 0, sin(theta_1), 0,            //
                  0,  0,  0,  0,  0,  0,  1, 0, 0, 0,                  0, 0, 0,                  0, 0, 0,            0,            //
                  0,  0,  0,  0,  0,  0,  0, 0, 0, 0,                  0, 0, 0,                  0, 0, 1,            0,            //
                  0,  0,  0,  0,  0,  0,  0, 0, 0, 1,                  0, 0, 0,                  0, 0, 0,            0,            //
                  1,  0,  0,  0,  0,  0,  0, 0, 0, 0,                  0, 0, -1*r2*sin(theta_2), 0, 0, 0,            cos(theta_2), //
                  0,  0,  0,  1,  0,  0,  0, 0, 0, 0,                  0, 0, r2*cos(theta_2),    0, 0, 0,            sin(theta_2), //
                  0,  0,  0,  0,  0,  0,  0, 1, 0, 0,                  0, 0, 0,                  0, 0, 0,            0,            //
                  0,  0,  0,  0,  0,  0,  0, 0, 0, 0,                  0, 0, 0,                  0, 0, 0,            1,            //
                  0,  0,  0,  0,  0,  0,  0, 0, 0, 0,                  0, 0, 1,                  0, 0, 0,            0;            //
            return H2 * _X;
        }
        Vm h(const Vn &_X) {
            State _state = get_state(_X);
            double r = _state.r;
            double theta = _state.yaw;
            // x_m = x + r*cos(theta)
            // y_m = y + r*sin(theta)
            H << 1,  0,  0,  0,  0,  0,  0, 0, 0, -1*r*sin(theta), 0, 0, 0, 0, 0, cos(theta), 0, //
                 0,  0,  0,  1,  0,  0,  0, 0, 0, r*cos(theta),    0, 0, 0, 0, 0, sin(theta), 0, //
                 0,  0,  0,  0,  0,  0,  1, 0, 0, 0,               0, 0, 0, 0, 0, 0,          0, //
                 0,  0,  0,  0,  0,  0,  0, 0, 0, 1,               0, 0, 0, 0, 0, 0,          0; //
            return H * _X;
        }
        
        // 先后融合两个测量值
        Vm h_after(const Vn &_X) {
            State _state = get_state(_X);
            double r = _state.r2;
            double theta = _state.yaw2;
            H <<  1,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, -1*r*sin(theta), 0, 0, 0, 0, cos(theta), 0, //
                  0,  0,  0,  1,  0,  0,  0, 0, 0, 0, 0, 0, r*cos(theta),    0, 0, 0, 0, sin(theta), 0, //
                  0,  0,  0,  0,  0,  0,  0, 1, 0, 0, 0, 0, 0,               0, 0, 0, 0, 0,          0, //
                  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 1,               0, 0, 0, 0, 0,          0; //
            return H * _X;
        }

        State predict(const double &dT) {
            f(dT);
            double  d_ax = X_prior[2] - X_posterior[2],
                    d_ay = X_prior[5] - X_posterior[5],
                    d_aw1 = X_prior[11] - X_posterior[11],
                    d_aw2 = X_prior[14] - X_posterior[14],
                    d_vz = X_prior[8] - X_posterior[8],
                    d_r1 = X_prior[15] - X_posterior[15],
                    d_r2 = X_prior[16] - X_posterior[16];
            double  q_x_x     =  1.0/36 * d_ax * d_ax * dT * dT * dT * dT, //
                    q_x_vx    =  1.0/12 * d_ax * d_ax * dT * dT * dT, //
                    q_x_ax    =  1.0/6 * d_ax * d_ax * dT * dT, //
                    q_vx_vx   =  1.0/4 * d_ax * d_ax * dT * dT, //
                    q_vx_ax   =  1.0/2 * d_ax * d_ax * dT, //
                    q_ax_ax   =  d_ax * d_ax, //
                    q_y_y     =  1.0/36 * d_ay * d_ay * dT * dT * dT * dT, //
                    q_y_vy    =  1.0/12 * d_ay * d_ay * dT * dT * dT, //
                    q_y_ay    =  1.0/6 * d_ay * d_ay * dT * dT, //
                    q_vy_vy   =  1.0/4 * d_ay * d_ay * dT * dT, //
                    q_vy_ay   =  1.0/2 * d_ay * d_ay * dT, //
                    q_ay_ay   =  d_ay * d_ay, //
                    q1_t_t    =  1.0/36 * d_aw1 * d_aw1 * dT * dT * dT * dT, //
                    q1_t_w    =  1.0/12 * d_aw1 * d_aw1 * dT * dT * dT, //
                    q1_t_aw   =  1.0/6 * d_aw1 * d_aw1 * dT * dT, //
                    q1_w_w    =  1.0/4 * d_aw1 * d_aw1 * dT * dT, //
                    q1_w_aw   =  1.0/2 * d_aw1 * d_aw1 * dT, //
                    q1_aw_aw  =  d_aw1 * d_aw1, //
                    q2_t_t    =  1.0/36 * d_aw2 * d_aw2 * dT * dT * dT * dT, //
                    q2_t_w    =  1.0/12 * d_aw2 * d_aw2 * dT * dT * dT, //
                    q2_t_aw   =  1.0/6 * d_aw2 * d_aw2 * dT * dT, //
                    q2_w_w    =  1.0/4 * d_aw2 * d_aw2 * dT * dT, //
                    q2_w_aw   =  1.0/2 * d_aw2 * d_aw2 * dT, //
                    q2_aw_aw  =  d_aw1 * d_aw1, //
                    q_z       =  1.0/4 * d_vz * d_vz * dT * dT, //
                    q_vz      =  1.0/2 * d_vz * d_vz * dT, //
                    q_az      =  d_vz * d_vz, //
                    q_r1      =  d_r1 * d_r1, //
                    q_r2      =  d_r2 * d_r2; //
            Q_prior <<  q_x_x,  q_x_vx,  q_x_ax,   0.,     0.,      0.,       0.,    0.,   0.,    0.,      0.,      0.,       0.,      0.,      0.,       0.,   0.,  
                        q_x_vx, q_vx_vx, q_vx_ax,  0.,     0.,      0.,       0.,    0.,   0.,    0.,      0.,      0.,       0.,      0.,      0.,       0.,   0.,  
                        q_x_ax, q_vx_ax, q_ax_ax,  0.,     0.,      0.,       0.,    0.,   0.,    0.,      0.,      0.,       0.,      0.,      0.,       0.,   0.,  
                        0.,     0.,      0.,       q_y_y,  q_y_vy,  q_y_ay,   0.,    0.,   0.,    0.,      0.,      0.,       0.,      0.,      0.,       0.,   0.,  
                        0.,     0.,      0.,       q_y_vy, q_vy_vy, q_vy_ay,  0.,    0.,   0.,    0.,      0.,      0.,       0.,      0.,      0.,       0.,   0.,  
                        0.,     0.,      0.,       q_y_ay, q_vy_ay, q_ay_ay,  0.,    0.,   0.,    0.,      0.,      0.,       0.,      0.,      0.,       0.,   0.,  
                        0.,     0.,      0.,       0.,     0.,      0.,       q_z,   q_z,  q_vz,  0.,      0.,      0.,       0.,      0.,      0.,       0.,   0.,  
                        0.,     0.,      0.,       0.,     0.,      0.,       q_z,   q_z,  q_vz,  0.,      0.,      0.,       0.,      0.,      0.,       0.,   0.,  
                        0.,     0.,      0.,       0.,     0.,      0.,       q_vz,  q_vz, q_az,  0.,      0.,      0.,       0.,      0.,      0.,       0.,   0.,  
                        0.,     0.,      0.,       0.,     0.,      0.,       0.,    0.,   0.,    q1_t_t,  q1_t_w,  q1_t_aw,  0.,      0.,      0.,       0.,   0.,  
                        0.,     0.,      0.,       0.,     0.,      0.,       0.,    0.,   0.,    q1_t_w,  q1_w_w,  q1_w_aw,  0.,      0.,      0.,       0.,   0.,  
                        0.,     0.,      0.,       0.,     0.,      0.,       0.,    0.,   0.,    q1_t_aw, q1_w_aw, q1_aw_aw, 0.,      0.,      0.,       0.,   0.,  
                        0.,     0.,      0.,       0.,     0.,      0.,       0.,    0.,   0.,    0.,      0.,      0.,       q2_t_t,  q2_t_w,  q2_t_aw,  0.,   0.,  
                        0.,     0.,      0.,       0.,     0.,      0.,       0.,    0.,   0.,    0.,      0.,      0.,       q2_t_w,  q2_w_w,  q2_w_aw,  0.,   0.,  
                        0.,     0.,      0.,       0.,     0.,      0.,       0.,    0.,   0.,    0.,      0.,      0.,       q2_t_aw, q2_w_aw, q2_aw_aw, 0.,   0.,  
                        0.,     0.,      0.,       0.,     0.,      0.,       0.,    0.,   0.,    0.,      0.,      0.,       0.,      0.,      0.,       q_r1, 0.,  
                        0.,     0.,      0.,       0.,     0.,      0.,       0.,    0.,   0.,    0.,      0.,      0.,       0.,      0.,      0.,       0.,   q_r2;

            // Q_prior = alpha * Q_prior + (1-alpha) * Q_posterior;
            P_prior = F * P_posterior * F.transpose() + Q_prior;
            return get_state(X_prior);
        }

        void measure(const Vm &measurement) {
            Z = measurement;
            Z_prior = h(X_prior);

            // R_prior = (Z - Z_prior) * (Z - Z_prior).transpose() - H * P_prior * H.transpose();
            // R_prior = R_posterior * (1-alpha) + R_prior * alpha; // 测量误差为全局误差
        }
        void measure2(const Vm2 &measurement) {
            Z2 = measurement;
            Z2_prior = h2(X_prior);

            // R2_prior = (Z2 - Z2_prior) * (Z2 - Z2_prior).transpose() - H2 * P_prior * H2.transpose();
            // R2_prior = R2_posterior * (1-alpha) + R2_prior * alpha; // 适用于测量误差为全局误差
        }
        void measure_after(const Vm &measurement) {
            Z = measurement;
            Z_prior = h_after(X_prior);
            
            // R_prior = (Z - Z_prior) * (Z - Z_prior).transpose() - H * P_prior * H.transpose();
            // R_prior = R_posterior * (1-alpha) + R_prior * alpha; // 测量误差为全局误差
        }

        void correct() {
            K = P_prior * H.transpose() * (H * P_prior * H.transpose() + R_prior).inverse();
            X_posterior = X_prior + K * (Z - Z_prior);
            log("correct_X_posterior", X_posterior);
            Mmn H_prior = H;
            Z_posterior = h(X_posterior);
            // Mnn P_last_posterior = P_posterior;
            P_posterior = (Mnn::Identity() - K * H_prior) * P_prior; // + K * R_prior * K.transpose();
            // R_posterior = (Z - Z_posterior) * (Z - Z_posterior).transpose() - H_prior * P_posterior * H_prior.transpose() + 2 * H_prior * K * ((Z - Z_prior) * (Z - Z_prior).transpose()) - 2 * H_prior * K * H_prior * P_prior * H_prior.transpose();
            // Q_posterior = (X_posterior - X_prior) * (X_posterior - X_prior).transpose() + P_posterior - F * P_last_posterior * F.transpose();
        }
        void correct2() {
            K2 = P_prior * H2.transpose() * (H2 * P_prior * H2.transpose() + R2_prior).inverse();
            X_posterior = X_prior + K2 * (Z2 - Z2_prior);
            log("correct2_X_posterior", X_posterior);
            Mmn2 H_prior = H2;
            Z2_posterior = h2(X_posterior);
            // Mnn P_last_posterior = P_posterior;
            P_posterior = (Mnn::Identity() - K2 * H_prior) * P_prior; // + K2 * R2_prior * K2.transpose();
            // R2_posterior = (Z2 - Z2_posterior) * (Z2 - Z2_posterior).transpose() - H_prior * P_posterior * H_prior.transpose() + 2 * H_prior * K2 * ((Z2 - Z2_prior) * (Z2 - Z2_prior).transpose()) - 2 * H_prior * K2 * H_prior * P_prior * H_prior.transpose();
            // Q_posterior = (X_posterior - X_prior) * (X_posterior - X_prior).transpose() + P_posterior - F * P_last_posterior * F.transpose();
        }
        void correct_after() {
            K = P_prior * H.transpose() * (H * P_prior * H.transpose() + R_prior).inverse();
            X_posterior = X_prior + K * (Z - Z_prior);
            Mmn H_prior = H;
            Z_posterior = h_after(X_posterior);
            P_posterior = (Mnn::Identity() - K * H_prior) * P_prior; // + K * R_prior * K.transpose();
        }
        void update(const Vm &measurement, const double dT) {
            alpha = alpha / (alpha + b);
            X_posterior = get_X(state);
            predict(dT);
            measure(measurement);
            correct();
            state = get_state(X_posterior);
            //RCLCPP_INFO(get_logger(), "pos: (%ld,%ld)", X_posterior[0], X_posterior[2]);
        }
        void update2(const Vm2 &measurement, const double dT) {
            alpha = alpha / (alpha + b);
            X_posterior = get_X(state);
            predict(dT);
            X_prior = X_posterior;
            X_prior = X_posterior;
            measure2(measurement);
            correct();
            state = get_state(X_posterior);
        }
        void update2(const Vm &measurement1, const Vm &measurement2, const double dT) {
            X_posterior = get_X(state);
            predict(dT);
            measure(measurement1)
            correct();
            X_prior = X_posterior;
            P_prior = P_posterior;
            measure_after(measurement2);
            correct_after();
            state = get_state(X_posterior);
        }
        
        double get_move_spd() {
            return sqrt(X_posterior[1] * X_posterior[1] + X_posterior[3] * X_posterior[3]);
        }

        double get_rotate_spd() {
            return X_posterior[9];
        }

        // State
        Vn X_prior;
        Vn X_posterior;
        Mnn F;
        Vn w;
        Mnn W;
        Mnn Q_prior;
        Mnn Q_posterior;
        Mnn P_prior;
        Mnn P_posterior;
        // Measurement
        Vm2 Z2;
        Vm2 Z2_prior;
        Vm2 Z2_posterior;
        Mmn2 H2;
        Vm2 v2;
        Mmm2 V2;
        Mmm2 R2_prior;
        Mmm2 R2_posterior;
        Vm Z;
        Vm Z_prior;
        Vm Z_posterior;
        Mmn H;
        Vm v;
        Mmm V;
        Mmm R_prior;
        Mmm R_posterior;
        // Kalman Gain
        Mnm2 K2;
        Mnm K;


        // Settings
        double b; // 渐消因子
        double alpha; // 权重因子
        int cirCnt = 0;
        rclcpp::Logger logger;
        State state;
        double last_r;

};
*/

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
            : x(x_), vx(vx_), y(y_), vy(vy_), yaw(yaw_), vyaw(vyaw_), z(z_), vz(vz_), z2(z_), r(r_), r2(r2_) {}
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


class enemy_double_observer_EKF {
   public:
    // 带后缀2的名称用于表示同时观测两块装甲板时用到的类型或变量
    using Vn = Eigen::Matrix<double, 13, 1>;
    using Vm = Eigen::Matrix<double, 4, 1>;
    using Vm2 = Eigen::Matrix<double, 10, 1>;
    using Vm_pts = Eigen::Matrix<double, 8, 1>;
    using Vm_pts2 = Eigen::Matrix<double, 16, 1>;
    using Mnn = Eigen::Matrix<double, 13, 13>;
    using Mmm = Eigen::Matrix<double, 4, 4>;
    using Mmm2 = Eigen::Matrix<double, 10, 10>;
    using Mmm_pts = Eigen::Matrix<double, 8, 8>;
    using Mmm_pts2 = Eigen::Matrix<double, 16, 16>;
    using Mmn = Eigen::Matrix<double, 4, 13>;
    using Mmn2 = Eigen::Matrix<double, 10, 13>;
    using Mmn_pts = Eigen::Matrix<double, 8, 13>;
    using Mmn_pts2 = Eigen::Matrix<double, 16, 13>;
    using Mnm = Eigen::Matrix<double, 13, 4>;
    using Mnm2 = Eigen::Matrix<double, 13, 10>;
    using Mnm_pts = Eigen::Matrix<double, 13, 8>;
    using Mnm_pts2 = Eigen::Matrix<double, 13, 16>;

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
    struct Observe_pts {
        double x1;
        double y1;
        double x2;
        double y2;
        double x3;
        double y3;
        double x4;
        double y4;
        Observe_pts() {}
        Observe_pts(double x1_, double y1_, double x2_, double y2_, double x3_, double y3_, double x4_, double y4_) : x1(x1_), y1(y1_), x2(x2_), y2(y2_), x3(x3_), y3(y3_), x4(x4_), y4(y4_) {}
    };
    struct Observe_pts2 {
        double x1a;
        double x1b;
        double y1a;
        double y1b;
        double x2a;
        double x2b;
        double y2a;
        double y2b;
        double x3a;
        double x3b;
        double y3a;
        double y3b;
        double x4a;
        double x4b;
        double y4a;
        double y4b;
        Observe_pts2() {}
        Observe_pts2(double x1_, double y1_, double x2_, double y2_, double x3_, double y3_, double x4_, double y4_, double x1__, double y1__, double x2__, double y2__, double x3__, double y3__, double x4__, double y4__) : x1a(x1_), y1a(y1_), x2a(x2_), y2a(y2_), x3a(x3_), y3a(y3_), x4a(x4_), y4a(y4_), x1b(x1__), y1b(y1__), x2b(x2__), y2b(y2__), x3b(x3__), y3b(y3__), x4b(x4__), y4b(y4__){}
    };
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
    static inline Vm_pts get_Z(Observe_pts _observe) {
        return Vm_pts(_observe.x1, _observe.y1, _observe.x2, _observe.y2, _observe.x3, _observe.y3, _observe.x4, _observe.y4);
    }
    static inline Observe_pts get_observe(Vm_pts _Z) {
        return Observe_pts(_Z[0], _Z[1], _Z[2], _Z[3], _Z[4], _Z[5], _Z[6], _Z[7]);
    }
    static inline Vm_pts2 get_Z(Observe_pts2 _observe) {
        return Vm_pts2(_observe.x1a, _observe.y1a, _observe.x2a, _observe.y2a, _observe.x3a, _observe.y3a, _observe.x4a, _observe.y4a, _observe.x1b, _observe.y1b, _observe.x2b, _observe.y2b, _observe.x3b, _observe.y3b, _observe.x4b, _observe.y4b);
    }
    static inline Observe_pts2 get_observe(Vm_pts2 _Z) {
        return Observe_pts2(_Z[0], _Z[1], _Z[2], _Z[3], _Z[4], _Z[5], _Z[6], _Z[7], _Z[8], _Z[9], _Z[10], _Z[11], _Z[12], _Z[13], _Z[14], _Z[15]);
    }


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
    // 7、8、9 两块装甲板的相位、角速度
    // 10、11 两块装甲板的半径
    std::vector<Vn> sample_X;  // 预测
    Vn Xp;
    Mnn Pp;

    std::vector<Vm_pts> sample_Z;
    Vm_pts Zp;
    Mmm_pts Pzz;
    Mnm_pts Pxz;

    // 自适应参数
    inline static double R_XYZ, R_YAW;
    inline static double Q2_XYZ, Q2_YAW, Q2_R;

    Mnn Pe;
    Mnn Q;
    Mmm R;
    Mmm2 R2;
    Mmm_pts R_pts;
    Mnm K;
    Mnm2 K2;
    Mnm_pts K_pts;

    explicit enemy_double_observer_EKF(Position_Calculator *pc) : logger(rclcpp::get_logger("enemy_KF")) {
        sample_num = 2 * n;
        samples = std::vector<Vn>(sample_num);
        weights = std::vector<double>(sample_num);
        Pe = init_P.asDiagonal();
        now_state_phase = 0;
        pc_ptr.reset(pc);
        sample_X = std::vector<Vn>(sample_num);
    }

    explicit enemy_double_observer_EKF() : logger(rclcpp::get_logger("enemy_KF")) {
        sample_num = 2 * n;
        samples = std::vector<Vn>(sample_num);
        weights = std::vector<double>(sample_num);
        Pe = init_P.asDiagonal();
        now_state_phase = 0;
        sample_X = std::vector<Vn>(sample_num);
    }

    void reset(
        const Observe &observe,
        const double& _r1 = 0.2,
        const double& _r2 = 0.15) {
        Pe = init_P.asDiagonal();
        Xe << observe.x - _r1 * cos(observe.yaw), //
              0,  //
              observe.y - _r1 * sin(observe.yaw), //
              0, //
              observe.z, //
              0, //
              observe.z, //
              observe.yaw, //
              0, //
              observe.yaw + ((angle_normalize(state.yaw) > angle_normalize(state.yaw2)) ? 1 : -1) * M_PI_2, // 
              0, //
              _r1,
              _r2;
        state = get_state(Xe);
    }
    void reset2(const Observe2 &observe) {
        Pe = init_P.asDiagonal();
        Xe << (observe.x-observe.r*cos(observe.yaw) + observe.x2-observe.r2*cos(observe.yaw2))/2,
               0,
               (observe.y-observe.r*sin(observe.yaw) + observe.y2-observe.r2*sin(observe.yaw2))/2,
               0, observe.z, 0, observe.z2, observe.yaw, 0, observe.yaw2, 0, observe.r, observe.r2;
        state = get_state(Xe);
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

    Vm h(const Vn &X) {
        State _state = get_state(X);
        Observe _observe;
        _observe.yaw = _state.yaw;
        _observe.z = _state.z;
        _observe.x = _state.x + _state.r * cos(_state.yaw);
        _observe.y = _state.y + _state.r * sin(_state.yaw);
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

    Vm_pts h(const Vn &X, bool isBigArmor, bool isMain) {
        State _state = get_state(X);
        Observe_pts _observe;
        Eigen::Vector3d xyz;
        std::vector<cv::Point2d> armor_img;
        if (isMain) {
            xyz << _state.x + _state.r * cos(_state.yaw), _state.y + _state.r * sin(_state.yaw), _state.z;
            armor_img = pc_ptr->generate_armor_img(isBigArmor, -15.0, _state.yaw / M_PI * 180, xyz);
        } else {
            xyz << _state.x + _state.r2 * cos(_state.yaw2), _state.y + _state.r2 * sin(_state.yaw2), _state.z2;
            armor_img = pc_ptr->generate_armor_img(isBigArmor, -15.0, _state.yaw2 / M_PI * 180, xyz);
        }
        _observe.x1 = armor_img[0].x;
        _observe.y1 = armor_img[0].y;
        _observe.x2 = armor_img[1].x;
        _observe.y2 = armor_img[1].y;
        _observe.x3 = armor_img[2].x;
        _observe.y3 = armor_img[2].y;
        _observe.x4 = armor_img[3].x;
        _observe.y4 = armor_img[3].y;
        return get_Z(_observe);
    }

    Vm_pts2 h2(const Vn &X, bool isBigArmor) {
        State _state = get_state(X);
        Observe_pts2 _observe;
        Eigen::Vector3d xyz;
        xyz << _state.x + _state.r * cos(_state.yaw), _state.y + _state.r * sin(_state.yaw), _state.z;
        std::vector<cv::Point2d> armor_img = pc_ptr->generate_armor_img(isBigArmor, -15.0, _state.yaw / M_PI * 180, xyz);
        _observe.x1a = armor_img[0].x;
        _observe.y1a = armor_img[0].y;
        _observe.x2a = armor_img[1].x;
        _observe.y2a = armor_img[1].y;
        _observe.x3a = armor_img[2].x;
        _observe.y3a = armor_img[2].y;
        _observe.x4a = armor_img[3].x;
        _observe.y4a = armor_img[3].y;
        xyz << _state.x + _state.r2 * cos(_state.yaw2), _state.y + _state.r2 * sin(_state.yaw2), _state.z2;
        armor_img = pc_ptr->generate_armor_img(isBigArmor, -15.0, _state.yaw2 / M_PI * 180, xyz);
        _observe.x1b = armor_img[0].x;
        _observe.y1b = armor_img[0].y;
        _observe.x2b = armor_img[1].x;
        _observe.y2b = armor_img[1].y;
        _observe.x3b = armor_img[2].x;
        _observe.y3b = armor_img[2].y;
        _observe.x4b = armor_img[3].x;
        _observe.y4b = armor_img[3].y;

        return get_Z(_observe);
    }

    State predict(double dT) { return get_state(f(Xe, dT)); }

    void CKF_predict(double dT) {
        //根据dT计算自适应Q
        static double dTs[4];
        for (int i = 1; i < 4; ++i) dTs[i] = dTs[i - 1] * dT;
        double q_x_x = dTs[3] / 4 * Q2_XYZ, q_x_vx = dTs[2] / 2 * Q2_XYZ, q_vx_vx = dTs[1] * Q2_XYZ;
        double q_y_y = dTs[3] / 4 * Q2_YAW, q_y_vy = dTs[2] / 2 * Q2_YAW, q_vy_vy = dTs[1] * Q2_YAW;
        double q_r_r = dTs[3] / 4 * Q2_R, q_r_vr = dTs[2] / 2 * Q2_R, q_vr_vr = dTs[1] * Q2_R;
        Q << q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                  //
            q_x_vx, q_vx_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                   //
            0., 0., q_x_vx, q_vx_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., 0., 0., q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0.,                                                                   //
            0., 0., 0., 0., q_x_vx, q_vx_vx, q_x_vx, 0., 0., 0., 0., 0., 0.,                                                                          //
            0., 0., 0., 0., 0., q_x_vx, q_x_x, 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., 0., 0., 0., 0., 0., q_y_y, q_y_vy, 0., 0., 0., 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., q_y_vy, q_vy_vy, 0., 0., 0., 0.,  
            0., 0., 0., 0., 0., 0., 0., 0., 0., q_y_y, q_y_vy, 0., 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., 0., 0., q_y_vy,q_vy_vy, 0., 0.,  
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., q_r_r, 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., q_r_r; //

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
    void CKF_measure(bool isBigArmor, bool isMain) {
        sample_Z = std::vector<Vm_pts>(sample_num);  // 修正
        Zp = Vm_pts::Zero();
        for (int i = 0; i < sample_num; ++i) {
            sample_Z[i] = h(samples[i], isBigArmor, isMain);
            Zp += weights[i] * sample_Z[i];
        }

        Pzz = Mmm_pts::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pzz += weights[i] * (sample_Z[i] - Zp) * (sample_Z[i] - Zp).transpose();
        }
        Pzz += R_pts;
    }
    void CKF_correct(const Vm_pts &z) {
        Pxz = Mnm_pts::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pxz += weights[i] * (sample_X[i] - Xp) * (sample_Z[i] - Zp).transpose();
        }
        K_pts = Pxz * Pzz.inverse();

        Xe = Xp + K_pts * (z - Zp);
        Pe = Pp - K_pts * Pzz * K_pts.transpose();

        state = get_state(Xe);
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
    void CKF_update(const Vm_pts &z1, const Vm_pts &z2, bool isBigArmor, double dT) {
        Xe = get_X(state);
        PerfGuard perf_KF("KF");
        // // 根据dis计算自适应R
        Observe_pts _observe = get_observe(z1);
        Vm_pts R_vec;
        R_vec << abs(R_XYZ * _observe.x1), abs(R_XYZ * _observe.y1), abs(R_XYZ * _observe.x2), abs(R_XYZ * _observe.y2), abs(R_XYZ * _observe.x3), abs(R_XYZ * _observe.y3), abs(R_XYZ * _observe.x4), abs(R_XYZ * _observe.y4);
        R_pts = R_vec.asDiagonal();
        
        //根据dT计算自适应Q
        static double dTs[4];
        for (int i = 1; i < 4; ++i) dTs[i] = dTs[i - 1] * dT;
        double q_x_x = dTs[3] / 4 * Q2_XYZ, q_x_vx = dTs[2] / 2 * Q2_XYZ, q_vx_vx = dTs[1] * Q2_XYZ;
        double q_y_y = dTs[3] / 4 * Q2_YAW, q_y_vy = dTs[2] / 2 * Q2_YAW, q_vy_vy = dTs[1] * Q2_YAW;
        double q_r_r = dTs[3] / 4 * Q2_R, q_r_vr = dTs[2] / 2 * Q2_R, q_vr_vr = dTs[1] * Q2_R;
        Q << q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                  //
            q_x_vx, q_vx_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                   //
            0., 0., q_x_vx, q_vx_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., 0., 0., q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0.,                                                                   //
            0., 0., 0., 0., q_x_vx, q_vx_vx, q_x_vx, 0., 0., 0., 0., 0., 0.,                                                                          //
            0., 0., 0., 0., 0., q_x_vx, q_x_x, 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., 0., 0., 0., 0., 0., q_y_y, q_y_vy, 0., 0., 0., 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., q_y_vy, q_vy_vy, 0., 0., 0., 0.,  
            0., 0., 0., 0., 0., 0., 0., 0., 0., q_y_y, q_y_vy, 0., 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., 0., 0., q_y_vy,q_vy_vy, 0., 0.,  
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., q_r_r, 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., q_r_r; //

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

        SRCR_sampling(Xp, Pp);
        std::vector<Vm_pts> sample_Z = std::vector<Vm_pts>(sample_num);  // 修正
        Vm_pts Zp = Vm_pts::Zero();
        for (int i = 0; i < sample_num; ++i) {
            sample_Z[i] = h(samples[i], isBigArmor, true);
            Zp += weights[i] * sample_Z[i];
        }

        Mmm_pts Pzz = Mmm_pts::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pzz += weights[i] * (sample_Z[i] - Zp) * (sample_Z[i] - Zp).transpose();
        }
        Pzz += R_pts;
        Mnm_pts Pxz = Mnm_pts::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pxz += weights[i] * (sample_X[i] - Xp) * (sample_Z[i] - Zp).transpose();
        }
        K_pts = Pxz * Pzz.inverse();

        Xe = Xp + K_pts * (z1 - Zp);
        Pe = Pp - K_pts * Pzz * K_pts.transpose();

        state = get_state(Xe);

        // round 2

        SRCR_sampling(Xe, Pe);
        _observe = get_observe(z2);
        R_vec = Vm_pts::Zero();
        R_vec << abs(R_XYZ * _observe.x1), abs(R_XYZ * _observe.y1), abs(R_XYZ * _observe.x2), abs(R_XYZ * _observe.y2), abs(R_XYZ * _observe.x3), abs(R_XYZ * _observe.y3), abs(R_XYZ * _observe.x4), abs(R_XYZ * _observe.y4);
        R_pts = R_vec.asDiagonal();
        sample_Z = std::vector<Vm_pts>(sample_num);  // 修正
        Zp = Vm_pts::Zero();
        for (int i = 0; i < sample_num; ++i) {
            sample_Z[i] = h(samples[i], isBigArmor, false);
            Zp += weights[i] * sample_Z[i];
        }

        Pzz = Mmm_pts::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pzz += weights[i] * (sample_Z[i] - Zp) * (sample_Z[i] - Zp).transpose();
        }
        sample_X = samples;
        Pzz += R_pts;
        Pxz = Mnm_pts::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pxz += weights[i] * (sample_X[i] - Xe) * (sample_Z[i] - Zp).transpose();
        }
        K_pts = Pxz * Pzz.inverse();

        Xe = Xe + K_pts * (z2 - Zp);
        Pe = Pe - K_pts * Pzz * K_pts.transpose();
        state = get_state(Xe);

        // CKF_predict(dT);
        // SRCR_sampling(Xp, Pp);

        // CKF_measure(isBigArmor, true);
        // CKF_correct(z1);

        // // round 2
        // // reset
        // sample_X = samples;
        // Xp = Xe;
        // Pp = Pe;
        // _observe = get_observe(z2);
        // R_vec << abs(R_XYZ * _observe.x1), abs(R_XYZ * _observe.y1), abs(R_XYZ * _observe.x2), abs(R_XYZ * _observe.y2), abs(R_XYZ * _observe.x3), abs(R_XYZ * _observe.y3), abs(R_XYZ * _observe.x4), abs(R_XYZ * _observe.y4);
        // R_pts = R_vec.asDiagonal();
        // SRCR_sampling(Xp,Pp);
        // CKF_measure(isBigArmor, false);
        // CKF_correct(z2);
        
    }
    void CKF_update(const Vm_pts &z, bool isBigArmor, double dT) {
        Xe = get_X(state);
        PerfGuard perf_KF("KF");
        Observe_pts _observe = get_observe(z);
        // 根据dis计算自适应R
        Vm_pts R_vec;
        R_vec << abs(R_XYZ * _observe.x1), abs(R_XYZ * _observe.y1), abs(R_XYZ * _observe.x2), abs(R_XYZ * _observe.y2), abs(R_XYZ * _observe.x3), abs(R_XYZ * _observe.y3), abs(R_XYZ * _observe.x4), abs(R_XYZ * _observe.y4);
        R_pts = R_vec.asDiagonal();
        //根据dT计算自适应Q
        static double dTs[4];
        
        for (int i = 1; i < 4; ++i) dTs[i] = dTs[i - 1] * dT;
        double q_x_x = dTs[3] / 4 * Q2_XYZ, q_x_vx = dTs[2] / 2 * Q2_XYZ, q_vx_vx = dTs[1] * Q2_XYZ;
        double q_y_y = dTs[3] / 4 * Q2_YAW, q_y_vy = dTs[2] / 2 * Q2_YAW, q_vy_vy = dTs[1] * Q2_YAW;
        double q_r_r = dTs[3] / 4 * Q2_R, q_r_vr = dTs[2] / 2 * Q2_R, q_vr_vr = dTs[1] * Q2_R;
        Q << q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                  //
            q_x_vx, q_vx_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                   //
            0., 0., q_x_vx, q_vx_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., 0., 0., q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0.,                                                                   //
            0., 0., 0., 0., q_x_vx, q_vx_vx, q_x_vx, 0., 0., 0., 0., 0., 0.,                                                                          //
            0., 0., 0., 0., 0., q_x_vx, q_x_x, 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., 0., 0., 0., 0., 0., q_y_y, q_y_vy, 0., 0., 0., 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., q_y_vy, q_vy_vy, 0., 0., 0., 0.,  
            0., 0., 0., 0., 0., 0., 0., 0., 0., q_y_y, q_y_vy, 0., 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., 0., 0., q_y_vy,q_vy_vy, 0., 0.,  
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., q_r_r, 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., q_r_r; //

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

        std::vector<Vm_pts> sample_Z = std::vector<Vm_pts>(sample_num);  // 修正
        Vm_pts Zp = Vm_pts::Zero();
        for (int i = 0; i < sample_num; ++i) {
            sample_Z[i] = h(samples[i], isBigArmor, true);
            Zp += weights[i] * sample_Z[i];
        }

        Mmm_pts Pzz = Mmm_pts::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pzz += weights[i] * (sample_Z[i] - Zp) * (sample_Z[i] - Zp).transpose();
        }
        Pzz += R_pts;

        Mnm_pts Pxz = Mnm_pts::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pxz += weights[i] * (sample_X[i] - Xp) * (sample_Z[i] - Zp).transpose();
        }

        K_pts = Pxz * Pzz.inverse();

        Xe = Xp + K_pts * (z - Zp);
        Pe = Pp - K_pts * Pzz * K_pts.transpose();

        state = get_state(Xe);
    }

    void CKF_update(const Vm &z, double dT) {
        Xe = get_X(state);
        PerfGuard perf_KF("KF");
        Observe _observe = get_observe(z);
        // 根据dis计算自适应R
        Vm R_vec;
        R_vec << abs(R_XYZ * _observe.x), abs(R_XYZ * _observe.y), abs(R_XYZ * _observe.z), R_YAW;
        R = R_vec.asDiagonal();
        //根据dT计算自适应Q
        static double dTs[4];
        
        for (int i = 1; i < 4; ++i) dTs[i] = dTs[i - 1] * dT;
        double q_x_x = dTs[3] / 4 * Q2_XYZ, q_x_vx = dTs[2] / 2 * Q2_XYZ, q_vx_vx = dTs[1] * Q2_XYZ;
        double q_y_y = dTs[3] / 4 * Q2_YAW, q_y_vy = dTs[2] / 2 * Q2_YAW, q_vy_vy = dTs[1] * Q2_YAW;
        double q_r_r = dTs[3] / 4 * Q2_R, q_r_vr = dTs[2] / 2 * Q2_R, q_vr_vr = dTs[1] * Q2_R;
        Q << q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                  //
            q_x_vx, q_vx_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                   //
            0., 0., q_x_vx, q_vx_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., 0., 0., q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0.,                                                                   //
            0., 0., 0., 0., q_x_vx, q_vx_vx, q_x_vx, 0., 0., 0., 0., 0., 0.,                                                                          //
            0., 0., 0., 0., 0., q_x_vx, q_x_x, 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., 0., 0., 0., 0., 0., q_y_y, q_y_vy, 0., 0., 0., 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., q_y_vy, q_vy_vy, 0., 0., 0., 0.,  
            0., 0., 0., 0., 0., 0., 0., 0., 0., q_y_y, q_y_vy, 0., 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., 0., 0., q_y_vy,q_vy_vy, 0., 0.,  
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., q_r_r, 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., q_r_r; //

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
    }

    void CKF_update2(const Vm2 &z, double dT) {
        Xe = get_X(state);
        PerfGuard perf_KF("KF");
        // 根据dis计算自适应R
        Observe2 _observe = get_observe(z);
        Vm2 R_vec;
        R_vec << abs(R_XYZ * _observe.x), abs(R_XYZ * _observe.y), abs(R_XYZ * _observe.z), abs(R_XYZ * _observe.r), abs(R_YAW * _observe.yaw), abs(R_XYZ * _observe.x2), abs(R_XYZ * _observe.y2), abs(R_XYZ * _observe.z2), abs(R_XYZ * _observe.r2), abs(R_YAW * _observe.yaw2);
        R2 = R_vec.asDiagonal();
        SRCR_sampling(Xe, Pe);

        // 根据dT计算自适应Q
        static double dTs[4];
        for (int i = 1; i < 4; ++i) dTs[i] = dTs[i - 1] * dT;
        double q_x_x = dTs[3] / 4 * Q2_XYZ, q_x_vx = dTs[2] / 2 * Q2_XYZ, q_vx_vx = dTs[1] * Q2_XYZ;
        double q_y_y = dTs[3] / 4 * Q2_YAW, q_y_vy = dTs[2] / 2 * Q2_YAW, q_vy_vy = dTs[1] * Q2_YAW;
        double q_r_r = dTs[3] / 4 * Q2_R, q_r_vr = dTs[2] / 2 * Q2_R, q_vr_vr = dTs[1] * Q2_R;
        Q << q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                  //
            q_x_vx, q_vx_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                   //
            0., 0., q_x_vx, q_vx_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., 0., 0., q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0.,                                                                   //
            0., 0., 0., 0., q_x_vx, q_vx_vx, q_x_vx, 0., 0., 0., 0., 0., 0.,                                                                          //
            0., 0., 0., 0., 0., q_x_vx, q_x_x, 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., 0., 0., 0., 0., 0., q_y_y, q_y_vy, 0., 0., 0., 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., q_y_vy, q_vy_vy, 0., 0., 0., 0.,  
            0., 0., 0., 0., 0., 0., 0., 0., 0., q_y_y, q_y_vy, 0., 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., 0., 0., q_y_vy,q_vy_vy, 0., 0.,  
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., q_r_r, 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., q_r_r; //

        std::vector<Vn> sample_X = std::vector<Vn>(sample_num);  // 预测
        Vn Xp = Vn::Zero();
        for (int i = 0; i < sample_num; ++i) {
            sample_X[i] = f(samples[i], dT);
            Xp += weights[i] * sample_X[i];
        }
        // logger.info("ekf_dt: {} {} {}",dT,dT,dT);
        
        Mnn Pp = Mnn::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pp += weights[i] * (sample_X[i] - Xp) * (sample_X[i] - Xp).transpose();
        }
        Pp += Q;

        SRCR_sampling(Xp, Pp);

        std::vector<Vm2> sample_Z = std::vector<Vm2>(sample_num);  // 修正
        Vm2 Zp = Vm2::Zero();
        for (int i = 0; i < sample_num; ++i) {
            sample_Z[i] = h2(samples[i]);
            Zp += weights[i] * sample_Z[i];
        }

        Mmm2 Pzz = Mmm2::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pzz += weights[i] * (sample_Z[i] - Zp) * (sample_Z[i] - Zp).transpose();
        }
        Pzz += R2;

        Mnm2 Pxz = Mnm2::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pxz += weights[i] * (sample_X[i] - Xp) * (sample_Z[i] - Zp).transpose();
        }

        K2 = Pxz * Pzz.inverse();
        Xe = Xp + K2 * (z - Zp);
        Pe = Pp - K2 * Pzz * K2.transpose();
        state = get_state(Xe);
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

/*
class enemy_double_observer_EKF {
   public:
    // 带后缀2的名称用于表示同时观测两块装甲板时用到的类型或变量
    using Vn = Eigen::Matrix<double, 13, 1>;
    using Vm = Eigen::Matrix<double, 4, 1>;
    using Vm2 = Eigen::Matrix<double, 10, 1>;
    using Mnn = Eigen::Matrix<double, 13, 13>;
    using Mmm = Eigen::Matrix<double, 4, 4>;
    using Mmm2 = Eigen::Matrix<double, 10, 10>;
    using Mmn = Eigen::Matrix<double, 4, 13>;
    using Mmn2 = Eigen::Matrix<double, 10, 13>;
    using Mnm = Eigen::Matrix<double, 13, 4>;
    using Mnm2 = Eigen::Matrix<double, 13, 10>;

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



    inline static Vn init_P;
    static constexpr int n = 13;    // 状态个数
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
    // 4、5、6 主装甲板高度，副装甲板高度、z方向速度
    // 7、8、9 两块装甲板的相位、角速度
    // 10、11 两块装甲板的半径

    // 自适应参数
    inline static double R_XYZ, R_YAW;
    inline static double Q2_XYZ, Q2_YAW, Q2_R;

    Mnn Pe;
    Mnn Q;
    Mmm R;
    Mmm2 R2;
    Mnm K;
    Mnm2 K2;

    explicit enemy_double_observer_EKF() : logger(rclcpp::get_logger("enemy_KF")) {
        sample_num = 2 * n;
        samples = std::vector<Vn>(sample_num);
        weights = std::vector<double>(sample_num);
        Pe = init_P.asDiagonal();
        now_state_phase = 0;
    }

    void reset(
        const Observe &observe,
        const double& _r1 = 0.2,
        const double& _r2 = 0.15) {
        Pe = init_P.asDiagonal();
        Xe << observe.x - _r1 * cos(observe.yaw), //
              0,  //
              observe.y - _r1 * sin(observe.yaw), //
              0, //
              observe.z, //
              0, //
              observe.z, //
              observe.yaw, //
              0, //
              observe.yaw + ((angle_normalize(state.yaw) > angle_normalize(state.yaw2)) ? 1 : -1) * M_PI_2, // 
              0, //
              _r1,
              _r2;
        state = get_state(Xe);
    }
    void reset2(const Observe2 &observe) {
        Pe = init_P.asDiagonal();
        Xe << (observe.x-observe.r*cos(observe.yaw) + observe.x2-observe.r2*cos(observe.yaw2))/2,
                    0,
                    (observe.y-observe.r*sin(observe.yaw) + observe.y2-observe.r2*sin(observe.yaw2))/2,
                    0, observe.z, 0, observe.z2, observe.yaw, 0, observe.yaw2, 0, observe.r, observe.r2;
        state = get_state(Xe);
    }



    // void log(char *msg, const Vn &_X) {
        // RCLCPP_WARN(logger, "%s: %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, ", msg, _X[0], _X[1], _X[2], _X[3], _X[4], _X[5],
                    // _X[6], _X[7], _X[8], _X[9], _X[10], _X[11], _X[12]);
    // }
    // void log(char *msg, const Vm &_Z) { RCLCPP_WARN(logger, "%s: %lf, %lf, %lf, %lf", msg, _Z[0], _Z[1], _Z[2], _Z[3]); }
    // void log(char *msg, const Vm2 &_Z2) {
        // RCLCPP_WARN(logger, "%s: %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf", msg, _Z2[0], _Z2[1], _Z2[2], _Z2[3], _Z2[4], _Z2[5], _Z2[6],
                    // _Z2[7], _Z2[8], _Z2[9]);
    // }

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

    Vm h(const Vn &X) {
        State _state = get_state(X);
        Observe _observe;
        _observe.yaw = _state.yaw;
        _observe.z = _state.z;
        _observe.x = _state.x + _state.r * cos(_state.yaw);
        _observe.y = _state.y + _state.r * sin(_state.yaw);
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
        Observe _observe = get_observe(z);
        // 根据dis计算自适应R
        Vm R_vec;
        R_vec << abs(R_XYZ * _observe.x), abs(R_XYZ * _observe.y), abs(R_XYZ * _observe.z), R_YAW;
        R = R_vec.asDiagonal();
        //根据dT计算自适应Q
        static double dTs[4];
        
        for (int i = 1; i < 4; ++i) dTs[i] = dTs[i - 1] * dT;
        double q_x_x = dTs[3] / 4 * Q2_XYZ, q_x_vx = dTs[2] / 2 * Q2_XYZ, q_vx_vx = dTs[1] * Q2_XYZ;
        double q_y_y = dTs[3] / 4 * Q2_YAW, q_y_vy = dTs[2] / 2 * Q2_YAW, q_vy_vy = dTs[1] * Q2_YAW;
        double q_r_r = dTs[3] / 4 * Q2_R, q_r_vr = dTs[2] / 2 * Q2_R, q_vr_vr = dTs[1] * Q2_R;
        Q << q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                  //
            q_x_vx, q_vx_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                   //
            0., 0., q_x_vx, q_vx_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., 0., 0., q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0.,                                                                   //
            0., 0., 0., 0., q_x_vx, q_vx_vx, q_x_vx, 0., 0., 0., 0., 0., 0.,                                                                          //
            0., 0., 0., 0., 0., q_x_vx, q_x_x, 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., 0., 0., 0., 0., 0., q_y_y, q_y_vy, 0., 0., 0., 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., q_y_vy, q_vy_vy, 0., 0., 0., 0.,  
            0., 0., 0., 0., 0., 0., 0., 0., 0., q_y_y, q_y_vy, 0., 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., 0., 0., q_y_vy,q_vy_vy, 0., 0.,  
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., q_r_r, 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., q_r_r; //


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
    }

    void CKF_update2(const Vm2 &z, double dT) {
        Xe = get_X(state);
        PerfGuard perf_KF("KF");
        // 根据dis计算自适应R
        Observe2 _observe = get_observe(z);
        Vm2 R_vec;
        R_vec << abs(R_XYZ * _observe.x), abs(R_XYZ * _observe.y), abs(R_XYZ * _observe.z), abs(R_XYZ * _observe.r), abs(R_YAW * _observe.yaw), abs(R_XYZ * _observe.x2), abs(R_XYZ * _observe.y2), abs(R_XYZ * _observe.z2), abs(R_XYZ * _observe.r2), abs(R_YAW * _observe.yaw2);
        R2 = R_vec.asDiagonal();
        SRCR_sampling(Xe, Pe);

        // 根据dT计算自适应Q
        static double dTs[4];
        for (int i = 1; i < 4; ++i) dTs[i] = dTs[i - 1] * dT;
        double q_x_x = dTs[3] / 4 * Q2_XYZ, q_x_vx = dTs[2] / 2 * Q2_XYZ, q_vx_vx = dTs[1] * Q2_XYZ;
        double q_y_y = dTs[3] / 4 * Q2_YAW, q_y_vy = dTs[2] / 2 * Q2_YAW, q_vy_vy = dTs[1] * Q2_YAW;
        double q_r_r = dTs[3] / 4 * Q2_R, q_r_vr = dTs[2] / 2 * Q2_R, q_vr_vr = dTs[1] * Q2_R;
        Q << q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                  //
            q_x_vx, q_vx_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                   //
            0., 0., q_x_vx, q_vx_vx, 0., 0., 0., 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., 0., 0., q_x_x, q_x_vx, 0., 0., 0., 0., 0., 0., 0.,                                                                   //
            0., 0., 0., 0., q_x_vx, q_vx_vx, q_x_vx, 0., 0., 0., 0., 0., 0.,                                                                          //
            0., 0., 0., 0., 0., q_x_vx, q_x_x, 0., 0., 0., 0., 0., 0.,                                                                 //
            0., 0., 0., 0., 0., 0., 0., q_y_y, q_y_vy, 0., 0., 0., 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., q_y_vy, q_vy_vy, 0., 0., 0., 0.,  
            0., 0., 0., 0., 0., 0., 0., 0., 0., q_y_y, q_y_vy, 0., 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., 0., 0., q_y_vy,q_vy_vy, 0., 0.,  
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., q_r_r, 0.,                                                                   //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., q_r_r; //

        std::vector<Vn> sample_X = std::vector<Vn>(sample_num);  // 预测
        Vn Xp = Vn::Zero();
        for (int i = 0; i < sample_num; ++i) {
            sample_X[i] = f(samples[i], dT);
            Xp += weights[i] * sample_X[i];
        }
        // logger.info("ekf_dt: {} {} {}",dT,dT,dT);
        
        Mnn Pp = Mnn::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pp += weights[i] * (sample_X[i] - Xp) * (sample_X[i] - Xp).transpose();
        }
        Pp += Q;

        SRCR_sampling(Xp, Pp);

        std::vector<Vm2> sample_Z = std::vector<Vm2>(sample_num);  // 修正
        Vm2 Zp = Vm2::Zero();
        for (int i = 0; i < sample_num; ++i) {
            sample_Z[i] = h2(samples[i]);
            Zp += weights[i] * sample_Z[i];
        }

        Mmm2 Pzz = Mmm2::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pzz += weights[i] * (sample_Z[i] - Zp) * (sample_Z[i] - Zp).transpose();
        }
        Pzz += R2;
        Eigen::IOFormat CleanFmt(6, 0, ", ", "\n", "[", "]");

        Mnm2 Pxz = Mnm2::Zero();
        for (int i = 0; i < sample_num; ++i) {
            Pxz += weights[i] * (sample_X[i] - Xp) * (sample_Z[i] - Zp).transpose();
            std::cout << "weights[" << i <<"]\n" << sample_X[i] << std::endl;
            std::cout << "Xp" << i <<"\n" << Xp << std::endl;
        }
        K2 = Pxz * Pzz.inverse();

        Xe = Xp + K2 * (z - Zp);
        Pe = Pp - K2 * Pzz * K2.transpose();
        state = get_state(Xe);
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
*/

class enemy_KF_Balance {
   public:
    // 带后缀2的名称用于表示同时观测两块装甲板时用到的类型或变量
    using Vn = Eigen::Matrix<double, 9, 1>;
    using Vm = Eigen::Matrix<double, 4, 1>;
    using Vm2 = Eigen::Matrix<double, 10, 1>;
    using Mnn = Eigen::Matrix<double, 9, 9>;
    using Mmm = Eigen::Matrix<double, 4, 4>;
    using Mmm2 = Eigen::Matrix<double, 10, 10>;
    using Mmn = Eigen::Matrix<double, 4, 9>;
    using Mmn2 = Eigen::Matrix<double, 10, 9>;
    using Mnm = Eigen::Matrix<double, 9, 4>;
    using Mnm2 = Eigen::Matrix<double, 9, 10>;

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
            : x(x_), vx(vx_), y(y_), vy(vy_), yaw(yaw_), vyaw(vyaw_), z(z_), vz(vz_), z2(z2_), r(r_), r2(r2_) {}
        State(){};
    };

    inline static Vn init_P;
    static constexpr int n = 9;    // 状态个数
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

    static constexpr double const_r = 0.21;

    // 自适应参数
    inline static double R_XYZ, R_YAW;
    inline static double Q2_XYZ, Q2_YAW, Q2_R;

    Mnn Pe;
    Mnn Q;
    Mmm R;
    Mnm K;

    explicit enemy_KF_Balance() : logger(rclcpp::get_logger("enemy_KF")) {
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
        _state.r = const_r;
        _state.r2 = const_r;
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
        return result;
    }

    Vm h(const Vn &X) {
        Vm result = Vm::Zero();
        result[0] = X[0] + const_r * cos(X[4]);
        result[1] = X[2] + const_r * sin(X[4]);
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
        //    xc      v_xc    yc      v_yc    yaw      v_yaw    za     v_za   r
        Q << q_x_x, q_x_vx, 0, 0, 0, 0, 0, 0, 0,        //
            q_x_vx, q_vx_vx, 0, 0, 0, 0, 0, 0, 0,       //
            0, 0, q_x_x, q_x_vx, 0, 0, 0, 0, 0,         //
            0, 0, q_x_vx, q_vx_vx, 0, 0, 0, 0, 0,       //
            0, 0, 0, 0, q_y_y, q_y_vy, 0, 0, 0,         //
            0, 0, 0, 0, q_y_vy, q_vy_vy, 0, 0, 0,       //
            0, 0, 0, 0, 0, 0, q_x_x, q_x_vx, 0,         //
            0, 0, 0, 0, 0, 0, q_x_vx, q_vx_vx, q_x_vx,  //
            0, 0, 0, 0, 0, 0, 0, q_x_vx, q_x_x,         //

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