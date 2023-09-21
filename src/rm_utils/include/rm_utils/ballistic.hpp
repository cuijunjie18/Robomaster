#ifndef _RMCV_BALLISTIC_HPP
#define _RMCV_BALLISTIC_HPP

#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <Eigen/Core>

class ballistic {
    // ballistic calculate result
   public:
    struct BallisticTable {
#define TABLE_WIDTH 1000
#define TABLE_HEIGHT 1000
       public:
        bool valid = false;
        double l, k, m, v, g;  // 枪长，空气阻力，弹丸质量，弹速，重力加速度
        int v_max;             // 弹速上限
        double theta_l, theta_r, theta_d;  // 下界，上界，步长
        int x_n, y_n;
        double x_l, x_r, x_d;
        double y_l, y_r, y_d;
        double sol_theta[TABLE_HEIGHT][TABLE_WIDTH],
            sol_t[TABLE_HEIGHT][TABLE_WIDTH];  // 解得的角度和时间，时间为 -1. 表示无解
#undef TABLE_WIDTH
#undef TABLE_HEIGHT
        bool contain(int x, int y) const { return 0 <= x && x < x_n && 0 <= y && y < y_n; }
    };

    struct ballistic_params {
        bool params_found;
        bool ballistic_refresh;
        double yaw2gun_offset;
        double v9, v15, v16, v18, v30;     // 弹速
        double small_k, big_k;             // 空气阻力
        double l;                          // 枪长
        double theta_l, theta_r, theta_d;  // 仰角(上界、下界、步长)
        double x_l, x_r;                   // 距离(上界、下界)
        double y_l, y_r;                   // 高度(上界、下界)
        int x_n, y_n;                      // 步数(距离、高度)
        std::vector<double> stored_yaw_offset;
        std::vector<double> stored_pitch_offset;
        std::vector<std::vector<double>> stored_offset_S;
        std::string table_dir;
    };

    struct bullet_res {
        bool fail;  // fail to solve
        double pitch, yaw, t;
        bullet_res(double _p, double _y, double _t) : fail(false), pitch(_p), yaw(_y), t(_t) {}
        bullet_res() : fail(true), pitch(0.), yaw(0.), t(0.) {}
    };
    static bool dump_table(const BallisticTable &table, const std::string &path);
    static bool load_table(BallisticTable &table, const std::string &path);
    static ballistic_params declare_ballistic_params_with_node(rclcpp::Node *node,
                                                               const std::string &table_dir);

    ballistic(const ballistic_params &param, const rclcpp::Logger &logger_);
    ballistic(rclcpp::Node *node, const std::string &table_dir);
    void reinit_ballistic(const ballistic_params &params_);
    void refresh_velocity(bool isBig, double velocity);
    double calibrate_ballistic(double x, double y);    

    bullet_res final_ballistic(Eigen::Matrix<double, 3, 1> p);
    std::pair<double, double> table_ballistic(const BallisticTable *tar, double x, double y);
    bullet_res calculate_angle(Eigen::Matrix<double, 3, 1> p);
    bullet_res segmented_ballistic(const Eigen::Matrix<double, 3, 1> &p);

   private:
    double v, yaw_offset, pitch_offset; // pitch_offset 向下为正  yaw_offset 向左为正
    const double g = 9.832, PI = 3.14159265358979;
    Eigen::Matrix<double, 3, 1> offset_S; // 发射装置与相机的偏移量

    const std::vector<double> seg_x{1.395, 1.92, 2.73, 3.61, 5.07, 6.39};
    const std::vector<double> seg_s{-0.1, -0.07, -0.05, 0., 0.2, 0.3};

    rclcpp::Logger logger;
    ballistic_params params;
    std::string table9_path, table15_path, table16_path, table18_path, table30_path;
    BallisticTable Table9, Table15, Table16, Table18, Table30;
    BallisticTable *pTable = nullptr;
    void calculate_table(bool is_big, int v_max);
};

#endif