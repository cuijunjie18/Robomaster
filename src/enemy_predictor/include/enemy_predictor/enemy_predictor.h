#ifndef _ENEMY_PREDICTOR_RMCV_H
#define _ENEMY_PREDICTOR_RMCV_H

#include <rm_utils/common.h>
#include <rm_utils/data.h>

#include <rm_interfaces/msg/detection.hpp>
#include <rm_interfaces/msg/rm_imu.hpp>
#include <rm_interfaces/msg/rmrobot.hpp>
#include <rm_utils/Position_Calculator.hpp>
#include <rm_utils/ballistic.hpp>
#include <std_msgs/msg/float64.hpp>
// ROS
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>

#include <camera_info_manager/camera_info_manager.hpp>
#include <image_transport/image_transport.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// predictor
#include <enemy_predictor/ekf.h>

#include <enemy_predictor/enemy_kf.hpp>
#include <queue>

namespace enemy_predictor {
enum Status { Alive = 0, Absent, Lost };

struct EnemyPredictorParams {
    std::string detection_name;
    std::string robot_name;
    std::string target_frame;
    std::string camera_name;
    bool enable_imshow;
    bool debug;
    vision_mode mode;
    bool right_press;  // 按下右键
    bool lobshot;      // 吊射模式
    Robot_id_dji robot_id;
    int rmcv_id;

    // EKF参数
    armor_EKF::config armor_ekf_config;
    yaw_KF::config yaw_kf_config;
    // 传统方法感知陀螺/前哨战相关参数
    double census_period_min;
    double census_period_max;
    double anti_outpost_census_period;
    double anti_outpost_census_period_min;
    double timestamp_thresh;  // 反前哨站时间差阈值
    double top_pitch_thresh;  // 判定建筑顶端装甲板的pitch阈值
    double outpost_top_offset_dis;
    double outpost_top_offset_z;
    bool is_aim_top;
    // 装甲目标过滤/选择
    double sight_limit;         // 过滤距离过远装甲板
    double high_limit;          // 过滤距离过高过低的装甲板
    double size_limit;          // 按面积过滤装甲板（太小）
    double bound_limit;         // 过滤图像边缘的装甲板（单位为像素）
    double aspect_limit_big;    // 当大装甲板处于40度时宽高比 公式m*sin(40)/n
    double aspect_limit_small;  // 当小装甲板处于40度时宽高比
    double rm_pnp_aspect_limit_big;
    double rm_pnp_aspect_limit_small;
    double reset_time;         // 若在视野中消失 reset_time秒，认为目标丢失
    double size_ratio_thresh;  // 切换整车滤波跟踪装甲板的面积阈值/切换选择目标的面积阈值
    cv::Point2d collimation;   // 二维图像上的准星
    // 帧间匹配
    double interframe_dis_thresh;    // 两帧间装甲板的最大移动距离（用于帧间匹配）
    int id_inertia;                  // 摩尔投标编号过滤惯性帧数
    double robot_2armor_dis_thresh;  // 同一车上相邻两装甲板最远距离
    // 运动状态判断
    double rotate_thresh;           // 进入旋转状态阈值
    double rotate_exit;             // 退出旋转状态阈值
    double high_spd_rotate_thresh;  // 进入高速旋转状态阈值
    double high_spd_rotate_exit;    // 退出高速旋转状态阈值
    double move_thresh;             // 进入移动状态阈值(目前移动小陀螺判断)
    double move_exit;               // 退出移动状态阈值
    // 火控参数
    double dis_thresh_kill;                    // 普通步兵斩杀线距离（在dis_thresh_common内的慢速目标开启高射频）
    double low_spd_thresh;                     // 自动发射运动速度阈值
    double gimbal_error_dis_thresh;            // 自动发弹阈值，限制云台误差的球面意义距离
    double gimbal_error_dis_thresh_old;        // 自动发弹阈值，限制云台误差的球面意义距离
    double residual_thresh;                    // 均值残差平方阈值判断发弹
    double tangential_spd_thresh;              // 切向速度阈值
    double normal_spd_thresh;                  // 径向速度阈值
    double decel_delay_time;                   // 减速停火时间
    bool choose_enemy_without_autoaim_signal;  // 在没有收到右键信号的时候也选择目标（调试用)
    bool disable_auto_shoot;
    // 延迟参数
    double response_delay;  // 系统延迟(程序+通信+云台响应)
    double shoot_delay;     // 发弹延迟
};

class TargetArmor {
   public:
    // Position_Calculator PC_armor;
    Status status = Absent, last_status = Absent;
    double alive_ts = -1., first_ts = -1.;  // 上次Alive的时间戳, 首次出现时间戳
    double dis_2d = INFINITY;               // 在二维图像中距离准星的距离（仅Alive时有效）
    double area_2d = 0.;                    // 在二维图像中的面积（仅Alive时有效）
    int vote_cnt = 1;                       // 摩尔投票计数
    int id = -1;
    int yaw_round = 0;  // yaw定义为:世界坐标系下目标相对于车的yaw
    double last_yaw = 0;
    int ori_yaw_round = 0;  // yaw定义为:世界坐标系下目标相对于车的yaw
    double last_ori_yaw = 0;
    bool matched = false;  // 帧间匹配标志位（这个可以不用放在类里面）
    bool following = false;
    bool tracking_in_enemy = false;  // 正在enemy中被追踪
    bool sub_tracking_in_enemy = false;
    int phase_in_enemy;
    bool just_appear;
    void zero_crossing(double datum);
    Position_Calculator::pnp_result position_data;  // 位姿
    cv::Rect_<float> bounding_box;                  // 四个识别点的外接矩形

    armor_EKF kf;
    armor_EKF::Vy getpos_xyz() const;
    armor_EKF::Vy getpos_pyd() const;
    // 滤波器更新接口，内部使用pyd进行KF更新

    yaw_KF yaw_kf;

    void initpos_xyz(const Position_Calculator::pnp_result &new_pb, const double TS);
    void updatepos_xyz(Position_Calculator::pnp_result &new_pb, const double TS);
    double get_yaw() { return yaw_round * M_PI * 2 + getpos_pyd()[1]; }
    double get_yaw_spd() { return kf.Xe[4]; }
    TargetArmor() : status(Alive) {}
};

class EnemyPredictorNode;

class Enemy {
   public:
    bool is_rotate = false, is_high_spd_rotate = false, is_move = false;
    struct enemy_positions {
        Eigen::Vector3d center;               // 车体中心二维xy坐标
        std::vector<Eigen::Vector3d> armors;  // 四个装甲板的xyz坐标
        std::vector<double> armor_yaws;       // 每一个装甲板对应的yaw值
        enemy_positions() : armors(4), armor_yaws(4) {}
    };
    Filter common_rotate_spd = Filter(5), common_middle_dis, common_yaw_spd = Filter(10);
    Filter common_move_spd = Filter(5);
    Filter outpost_aiming_pos[3];
    Filter balance_judge;
    // Logger logger;
    EnemyPredictorNode *predictor;
    Status status = Status::Absent;
    double last_yaw;
    double last_yaw2;
    double yaw;
    double yaw2;
    double last_ob_r;
    double last_ob_r2;
    double last_r;
    double last_r2;
    double r;
    double r2;
    std::vector<double> r_data_set[2];
    std::vector<double> dz_data_set;
    std::vector<double> z_data_set[2];
    int yaw_round = 0;
    int yaw2_round = 0;
    double alive_ts = -1;
    double t_absent;  // 处于absent状态的时间
    double last_update_ekf_ts = -1;
    double dz = 0;
    int id = -1;
    bool armor_appr = false;
    bool enemy_kf_init = false;
    bool double_track = false;
    bool following = false;
    bool tracking_absent_flag = false;
    bool sub_tracking_absent_flag = false;
    double min_dis_2d = INFINITY;
    int armor_cnt = 4;
    double appr_period;
    std::deque<std::pair<double, double>> mono_inc, mono_dec;
    std::deque<std::pair<double, double>> TSP;
    std::vector<TargetArmor> armors;
    std::vector<Filter> armor_dis_filters;
    std::vector<Filter> armor_z_filters;
    void add_armor(TargetArmor &armor);
    void armor_appear(TargetArmor &armor);  // 出现新装甲板时调用，统计旋转信息

    enemy_KF_4 enemy_kf;
    enemy_positions predict_positions(double stamp);
    double ori_diff;
    double get_rotate_spd() { return enemy_kf.state.omega; }
    double get_move_spd() { return sqrt(enemy_kf.state.vx * enemy_kf.state.vx + enemy_kf.state.vy * enemy_kf.state.vy); }
    void update_motion_state();
    void set_unfollowed();
    explicit Enemy(EnemyPredictorNode *predictor_);
};
using IterEnemy = std::vector<Enemy>::iterator;

// 专门用于处理Enemy内部的armor情况
struct EnemyArmor {
    int phase;
    double yaw_distance_predict;
    Eigen::Vector3d pos;
};

struct match_edge {
    int last_enemy_idx;  // 匹配的对手下标
    int last_sub_idx;    // 匹配的对手装甲下标
    int now_idx;         // 自己的下标
    double match_dis;
    bool operator<(const match_edge &e) const { return match_dis < e.match_dis; }
    match_edge(int _enemy, int _armor, int _self, double _dis) : last_enemy_idx(_enemy), last_sub_idx(_armor), now_idx(_self), match_dis(_dis) {}
};

struct match_armor {
    Position_Calculator::pnp_result position;
    int armor_id;
    int detection_idx;  // 在detections中的下标
    bool isBigArmor;
    bool matched;
    cv::Rect_<float> bbox;
    match_armor(const Position_Calculator::pnp_result _position, int _id, int _idx, bool _big, bool _matched, cv::Rect_<float> _bbox)
        : position(_position), armor_id(_id), detection_idx(_idx), isBigArmor(_big), matched(_matched), bbox(_bbox) {}
};

struct detect_msg {
    double time_stamp;
    cv::Mat img;
    std::vector<Armor> res;
    vision_mode mode;
};

class EnemyPredictorNode : public rclcpp::Node {
   public:
    explicit EnemyPredictorNode(const rclcpp::NodeOptions &options);
    ~EnemyPredictorNode() override;

    EnemyPredictorParams params;
    detect_msg recv_detection;      // 保存识别到的目标信息
    rm_interfaces::msg::RmImu imu;  // 保存收到的IMU信息 //TODO
    std::vector<Enemy> enemies;
    std::shared_ptr<ballistic> bac;
    std::array<int, 9UL> enemy_armor_type;  // 敌方装甲板大小类型 1 大 0 小
    cv::Mat show_enemies;
    ControlMsg off_cmd;
    ControlMsg make_cmd(double roll, double pitch, double yaw, uint8_t flag, uint8_t follow_id);
    Position_Calculator pc;

   private:
    // 位姿解算与变换相关
    std::shared_ptr<tf2_ros::Buffer> tf2_buffer;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener;

    // pub/sub
    rclcpp::Subscription<rm_interfaces::msg::Detection>::SharedPtr detection_sub;
    rclcpp::Subscription<rm_interfaces::msg::Rmrobot>::SharedPtr robot_sub;
    rclcpp::Publisher<rm_interfaces::msg::Control>::SharedPtr control_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr show_enemies_pub;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pnp_pose_pub;
    std::vector<rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr> watch_data_pubs;
    void add_point_Marker(double x_, double y_, double z_, double r_, double g_, double b_, double a_, Eigen::Vector3d pos);
    visualization_msgs::msg::MarkerArray markers;
    int marker_id;

    bool is_big_armor(armor_type type);
    int get_armor_cnt(armor_type type);

    void update_armors();
    void update_enemy();

    IterEnemy select_enemy_oritation();
    ballistic::bullet_res center_ballistic(const IterEnemy &, double delay);
    ballistic::bullet_res calc_ballistic(const IterEnemy &, int armor_phase, double delay);
    EnemyArmor select_armor_directly(const IterEnemy &);
    ControlMsg get_command();

    // IterEnemy select_enemy_nearest2d();  // 选择enemy
    // IterEnemy select_enemy_lobshot();
    // TargetArmor &select_armor_old(const IterEnemy &);           // 考虑上次的目标，计算击打目标
    // TargetArmor &select_armor_directly_old(const IterEnemy &);  // 上次目标丢失时，计算击打目标
    // ballistic::bullet_res calc_ballistic(const armor_EKF &armor_kf, double delay);
    // ballistic::bullet_res center_ballistic(const IterEnemy &, double delay);
    double change_spd_ts = 0;

    void load_params();
    void get_params();
    rclcpp::TimerBase::SharedPtr params_timer;
    void detection_callback(rm_interfaces::msg::Detection::UniquePtr detection_msg);
    void robot_callback(rm_interfaces::msg::Rmrobot::SharedPtr robot_msg);
};
};  // namespace enemy_predictor

#endif