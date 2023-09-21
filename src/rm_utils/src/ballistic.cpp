#include <yaml-cpp/yaml.h>

#include <cstdio>
#include <fstream>
#include <rm_utils/ballistic.hpp>
#include <string>

ballistic::ballistic(const ballistic_params &param, const rclcpp::Logger &logger_): logger(logger_){
    reinit_ballistic(param);
}
ballistic::ballistic(rclcpp::Node *node, const std::string &table_dir):logger(node->get_logger()){
    params = declare_ballistic_params_with_node(node,table_dir);
    reinit_ballistic(params);
}


ballistic::ballistic_params ballistic::declare_ballistic_params_with_node(
    rclcpp::Node *node, const std::string &table_dir) {
    ballistic_params params;
    params.params_found = node->declare_parameter("ballistic.params_found", false);
    // params_found为false说明加载失败
    assert(params.params_found && "Cannot found valid ballistic params");
    params.yaw2gun_offset = node->declare_parameter("ballistic.yaw2gun_offset", 0.00);
    params.stored_yaw_offset =
        node->declare_parameter("ballistic.stored_yaw_offset", std::vector<double>({}));
    params.stored_pitch_offset =
        node->declare_parameter("ballistic.stored_pitch_offset", std::vector<double>({}));
    std::string offset_s_string = node->declare_parameter("ballistic.stored_offset_S", "");
    // 使用yaml-cpp 处理string
    YAML::Node v = YAML::Load(offset_s_string);
    params.stored_offset_S = v.as<std::vector<std::vector<double>>>();

    assert(params.stored_yaw_offset.size() == 5 && "stored_yaw_offset size must be 5!");
    assert(params.stored_pitch_offset.size() == 5 && "stored_pitch_offset size must be 5!");
    assert(params.stored_offset_S.size() == 5 && "stored_offset_S size must be 5!");
    params.ballistic_refresh = node->declare_parameter("ballistic.ballistic_refresh", false);
    params.small_k = node->declare_parameter("ballistic.small_k", 0.0);
    params.big_k = node->declare_parameter("ballistic.big_k", 0.0);
    params.l = node->declare_parameter("ballistic.l", 0.0);
    params.theta_l = node->declare_parameter("ballistic.theta_l", 0.0);
    params.theta_r = node->declare_parameter("ballistic.theta_r", 0.0);
    params.theta_d = node->declare_parameter("ballistic.theta_d", 0.0);
    params.x_l = node->declare_parameter("ballistic.x_l", 0.0);
    params.x_r = node->declare_parameter("ballistic.x_r", 0.0);
    params.x_n = node->declare_parameter("ballistic.x_n", 0);
    params.y_l = node->declare_parameter("ballistic.y_l", 0.0);
    params.y_r = node->declare_parameter("ballistic.y_r", 0.0);
    params.y_n = node->declare_parameter("ballistic.y_n", 0);
    params.v9 = node->declare_parameter("ballistic.v9", 0.0);
    params.v16 = node->declare_parameter("ballistic.v16", 0.0);
    params.v15 = node->declare_parameter("ballistic.v15", 0.0);
    params.v18 = node->declare_parameter("ballistic.v18", 0.0);
    params.v30 = node->declare_parameter("ballistic.v30", 0.0);
    params.table_dir = table_dir;
    bool print_params = node->declare_parameter("ballistic.print_params", false);
    if (print_params) {
        RCLCPP_INFO(node->get_logger(),
                    "Ballistic velocities: v9:%.2f v16:%.2f / v15:%.2f v:18%.2f v30:%.2f",
                    params.v9, params.v16, params.v15, params.v18, params.v30);
    }
    return params;
}

void ballistic::reinit_ballistic(const ballistic::ballistic_params &params_) {
    params = params_;
    table9_path = params.table_dir + "/" + "big_9.dat";
    table16_path = params.table_dir + "/" + "big_16.dat";
    table15_path = params.table_dir + "/" + "small_15.dat";
    table18_path = params.table_dir + "/" + "small_18.dat";
    table30_path = params.table_dir + "/" + "small_30.dat";
    if (params.ballistic_refresh) {
        calculate_table(true, 9);
        calculate_table(true, 16);
        calculate_table(false, 15);
        calculate_table(false, 18);
        calculate_table(false, 30);
    }
    if (!load_table(Table9, table9_path)) {
        RCLCPP_WARN(logger, "Ballistic Table 9 load failed.");
        calculate_table(true, 9);
        if (!load_table(Table9, table9_path)) {
            RCLCPP_WARN(logger, "Ballistic Table 9 reload failed.");
        }
    }
    if (!load_table(Table16, table16_path)) {
        RCLCPP_WARN(logger, "Ballistic Table 16 load failed.");
        calculate_table(true, 16);
        if (!load_table(Table16, table16_path)) {
            RCLCPP_WARN(logger, "Ballistic Table 16 reload failed.");
        }
    }
    if (!load_table(Table15, table15_path)) {
        RCLCPP_WARN(logger, "Ballistic Table 15 load failed.");
        calculate_table(false, 15);
        if (!load_table(Table15, table16_path)) {
            RCLCPP_WARN(logger, "Ballistic Table 15 reload failed.");
        }
    }
    if (!load_table(Table18, table18_path)) {
        RCLCPP_WARN(logger, "Ballistic Table 18 load failed.");
        calculate_table(false, 18);
        if (!load_table(Table18, table16_path)) {
            RCLCPP_WARN(logger, "Ballistic Table 18 reload failed.");
        }
    }
    if (!load_table(Table30, table30_path)) {
        RCLCPP_WARN(logger, "Ballistic Table 30 load failed.");
        calculate_table(false, 30);
        if (!load_table(Table30, table16_path)) {
            RCLCPP_WARN(logger, "Ballistic Table 30 reload failed.");
        }
    }
}

bool ballistic::dump_table(const ballistic::BallisticTable &table, const std::string &path) {
    std::ofstream fout(path, std::ofstream::binary);
    if (!fout) return false;
    fout.write(reinterpret_cast<const char *>(&table), sizeof(table));
    fout.close();
    return true;
}

bool ballistic::load_table(ballistic::BallisticTable &table, const std::string &path) {
    std::ifstream fin(path, std::ifstream::binary);
    if (!fin) return false;
    fin.read(reinterpret_cast<char *>(&table), sizeof(table));
    fin.close();
    return true;
}

// 输入为弧度角，判断当前两角的角度差距是否小于0.1角度
inline bool feq(double a, double b) { return fabs(a - b) * 180 / M_PI < 0.1; }

void ballistic::calculate_table(bool is_big, int v_max) {
    // Logger logger("calculate_ballistic");

    std::shared_ptr<BallisticTable> tar = std::make_shared<BallisticTable>();
    tar->v_max = v_max;
    tar->g = 9.7925;
    if (is_big)
        tar->m = 0.0445;
    else
        tar->m = 0.0032;
    if (is_big)
        tar->k = params.big_k;  // 空气阻力
    else
        tar->k = params.small_k;
    tar->v = 16;  // 弹速
    if (v_max == 30)
        tar->v = params.v30;
    else if (v_max == 18)
        tar->v = params.v18;
    else if (v_max == 16)
        tar->v = params.v16;
    else if (v_max == 15)
        tar->v = params.v15;
    else if (v_max == 9)
        tar->v = params.v9;
    tar->l = params.l;              // 发射点（摩擦轮）到云台pitch转轴的距离
    tar->theta_l = params.theta_l;  // 仰角下界
    tar->theta_r = params.theta_r;  // 仰角上界
    tar->theta_d = params.theta_d;  // 仰角步长
    tar->x_l = params.x_l;          // 距离下界
    tar->x_r = params.x_r;          // 距离上界
    tar->x_n = params.x_n;          // 距离步数
    tar->y_l = params.y_l;          // 高度下界
    tar->y_r = params.y_r;          // 高度上界
    tar->y_n = params.y_n;          // 高度步数
    RCLCPP_INFO(logger, "ballistic:H");
    for (int i = 0; i < tar->x_n; i++) std::fill_n(tar->sol_t[i], tar->y_n, -1);
    // 用于判断当前轨迹的距离是否适合覆盖角点（如果在差距这个范围内，则视为可覆盖）
    double thresh_cover_rate = 1.2;
    double x_d = (tar->x_r - tar->x_l) / tar->x_n;
    double y_d = (tar->y_r - tar->y_l) / tar->y_n;
    tar->x_d = x_d;
    tar->y_d = y_d;
    // 枚举子弹发射的角度（从大到小，后面枚举的角度会在交叉轨迹处覆盖前面枚举的角度，优先选择角度更小的）
    for (double theta = tar->theta_r; theta >= tar->theta_l; theta -= tar->theta_d) {
        double theta_rad = theta * M_PI / 180;
        double co = cos(theta_rad);
        for (int i = 0; i < tar->x_n; i++) {  // 枚举子弹飞行到的x坐标
            double x = tar->x_l + x_d * i;
            double tra_x = x - tar->l * co;
            double t = -1;
            if (1 - tar->k / (tar->m * tar->v * co) * tra_x > 0) {
                t = -tar->m / tar->k * log(1 - tar->k / (tar->m * tar->v * co) * tra_x);
            } else {
                continue;
            }
            double tra_y = (tan(theta_rad) + tar->m * tar->g / (tar->k * tar->v * co)) * tra_x -
                           tar->m * tar->g / tar->k * t;
            double y = tra_y + tar->l * sin(theta_rad);  // 通过解微分方程，推导出子弹飞行到的y坐标
            double j = (y - tar->y_l) / y_d;  // 当前y值对应的弹道表的列坐标
            int j_fl = floor(j), j_ce = ceil(j);
            if (j_fl >= 0 && j_fl < tar->y_n && j - j_fl <= thresh_cover_rate) {
                if (tar->sol_t[i][j_fl] < 0. || !feq(tar->sol_theta[i][j_fl], theta_rad)) {
                    tar->sol_theta[i][j_fl] = theta_rad;
                    tar->sol_t[i][j_fl] = t;
                }
            }
            if (j_ce >= 0 && j_ce < tar->y_n && j_ce - j <= thresh_cover_rate) {
                if (tar->sol_t[i][j_ce] < 0. || !feq(tar->sol_theta[i][j_ce], theta_rad)) {
                    tar->sol_theta[i][j_ce] = theta_rad;
                    tar->sol_t[i][j_ce] = t;
                }
            }
        }
    }
    for (int i = 0; i < tar->x_n; i++) {
        int last = -1;
        for (int j = 0; j < tar->y_n; j++) {
            if (tar->sol_t[i][j] > 0) {
                if (last == -1)
                    last = j;
                else {  // 这个做法实际上不好、可能会存在结尾处没有轨迹覆盖的点，就无法使用一维线性插值进行计算
                    double delta_theta =
                        (tar->sol_theta[i][j] - tar->sol_theta[i][last]) / (j - last);
                    double delta_t = (tar->sol_t[i][j] - tar->sol_t[i][last]) / (j - last);
                    for (int k = last + 1; k < j; k++) {
                        tar->sol_theta[i][k] = tar->sol_theta[i][last] + delta_theta * (k - last);
                        tar->sol_t[i][k] = tar->sol_t[i][last] + delta_t * (k - last);
                    }
                }
                last = j;
            }
        }
    }
    tar->valid = true;
    // char buffer[200] = "";
    // sprintf(buffer, "%sballistic/%s_%d.dat", ROOT, is_big ? "big" : "small", tar->v_max);
    std::string save_buffer =
        params.table_dir + "/" + (is_big ? "big" : "small") + "_" + std::to_string(tar->v_max) + ".dat";
    dump_table(*tar, save_buffer);
    return;
}

void ballistic::refresh_velocity(bool isBig, double velocity) {
    v = velocity;
    if (isBig) {
        if (v < 9.) {
            pTable = &Table9;
            offset_S = Eigen::Vector3d(params.stored_offset_S[0].data());
            yaw_offset = params.stored_yaw_offset[0];
            pitch_offset = params.stored_pitch_offset[0];
        } else {
            pTable = &Table16;
            offset_S = Eigen::Vector3d(params.stored_offset_S[2].data());
            yaw_offset = params.stored_yaw_offset[2];
            pitch_offset = params.stored_pitch_offset[2];
        }
    } else {
        if (v < 15.) {
            pTable = &Table15;
            offset_S = Eigen::Vector3d(params.stored_offset_S[1].data());
            yaw_offset = params.stored_yaw_offset[1];
            pitch_offset = params.stored_pitch_offset[1];
        } else if (v < 18.) {
            pTable = &Table18;
            offset_S = Eigen::Vector3d(params.stored_offset_S[3].data());
            yaw_offset = params.stored_yaw_offset[3];
            pitch_offset = params.stored_pitch_offset[3];
        } else {
            pTable = &Table30;
            offset_S = Eigen::Vector3d(params.stored_offset_S[4].data());
            yaw_offset = params.stored_yaw_offset[4];
            pitch_offset = params.stored_pitch_offset[4];
        }
    }
}

double ballistic::calibrate_ballistic(double x, double y) {
    RCLCPP_INFO(logger, "calibrate_ballistic: x:%lf y:%lf", x, y);
    auto [pitch, t] = table_ballistic(
        pTable, sqrt(x * x - params.yaw2gun_offset * params.yaw2gun_offset), y + offset_S[2]);
    return (-pitch + pitch_offset * PI / 180);
}

std::pair<double, double> ballistic::table_ballistic(const ballistic::BallisticTable *tar, double x,
                                                     double y) {
    int i = floor((x - tar->x_l) / tar->x_d), j = floor((y - tar->y_l) / tar->y_d);
    if (tar->contain(i, j) && tar->contain(i + 1, j + 1)) {  // 判断在表中存在
        // 二维线性插值
        double f00 = tar->sol_theta[i][j], f01 = tar->sol_theta[i][j + 1],
               f10 = tar->sol_theta[i + 1][j];  // 在表中取三个邻点
        double t00 = tar->sol_t[i][j], t01 = tar->sol_t[i][j + 1], t10 = tar->sol_t[i + 1][j];
        double df_dx = (f10 - f00) / tar->x_d,
               df_dy = (f01 - f00) / tar->y_d;  // 计算偏微分
        double dt_dx = (t10 - t00) / tar->x_d, dt_dy = (t01 - t00) / tar->y_d;
        double delta_x = x - i * tar->x_d - tar->x_l, delta_y = y - j * tar->y_d - tar->y_l;
        return std::make_pair(f00 + df_dx * delta_x + df_dy * delta_y,
                              t00 + dt_dx * delta_x + dt_dy * delta_y);  // 用全微分近似全增量
    } else {
        return std::make_pair(0., -1.);
    }
}

ballistic::bullet_res ballistic::final_ballistic(Eigen::Matrix<double, 3, 1> p) {
    // Logger logger("final_ballistic");
    double x = sqrt(p[0] * p[0] + p[1] * p[1]), y = p[2];
    // logger.info("x:{} y:{}", x, y);
    RCLCPP_INFO(logger, "final_ballistic: x:%lf y:%lf", x, y);
    auto [pitch, t] = table_ballistic(
        pTable, sqrt(x * x - params.yaw2gun_offset * params.yaw2gun_offset), y + offset_S[2]);
    if (t < 0) return bullet_res();
    return bullet_res(-pitch + pitch_offset * PI / 180,
                      atan2(p[1], p[0]) + asin(params.yaw2gun_offset / x) + yaw_offset * PI / 180,
                      t);
}

ballistic::bullet_res ballistic::segmented_ballistic(const Eigen::Matrix<double, 3, 1> &p) {
    double x = sqrt(p[0] * p[0] + p[1] * p[1]);
    for (int i = 0; i + 1 < (int)seg_x.size(); ++i) {
        if (seg_x[i] <= x && x <= seg_x[i + 1]) {
            offset_S[2] =
                seg_s[i] + (seg_s[i + 1] - seg_s[i]) * (x - seg_x[i]) / (seg_x[i + 1] - seg_x[i]);
            if (p[2] > -0.3) {
                offset_S[2] += 0.04;
            }
            return final_ballistic(p);
        }
    }
    return bullet_res{-1e9, -1e9, -1e9};
}