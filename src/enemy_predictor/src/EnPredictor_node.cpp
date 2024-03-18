#include <enemy_predictor/ekf.h>
#include <enemy_predictor/enemy_predictor.h>

#include <Eigen/Core>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <functional>
using namespace enemy_predictor;

Enemy::Enemy(EnemyPredictorNode *predictor_, int _id, bool _enemy_kf_init, bool _following, int _armor_cnt)
    : id(_id), enemy_kf_init(_enemy_kf_init), following(_following), armor_cnt(_armor_cnt), predictor(predictor_), enemy_kf(predictor_) {
    for (int i = 0; i < 3; ++i) {
        outpost_aiming_pos[i] = Filter(1000);
    }
    armor_dis_filters = std::vector<Filter>(4, Filter(100, HarmonicMean));
    armor_z_filters = std::vector<Filter>(4, Filter(100, ArithmeticMean));
    armor_disyaw_mean_filters = std::vector<Filter>(armor_cnt, Filter(30, ArithmeticMean));
    armor_disyaw_mean2_filters = std::vector<Filter>(armor_cnt, Filter(30, ArithmeticMean));
    for (int i = 0; i < 4; ++i) {
        armor_dis_filters[i].update(0.2);
        armor_z_filters[i].update(-0.1);
    }
    armors_yaw_history.resize(armor_cnt);
    armors_disyaw_llimit.resize(armor_cnt);
    armors_disyaw_rlimit.resize(armor_cnt);
}

EnemyPredictorNode::EnemyPredictorNode(const rclcpp::NodeOptions &options) : Node("enemy_predictor", options) {
    RCLCPP_INFO(get_logger(), "EnemyPredictor Start!");
    load_params();

    params_timer = this->create_wall_timer(std::chrono::seconds(2), std::bind(&EnemyPredictorNode::get_params, this));

    // tf2 relevant
    tf2_buffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    // Create the timer interface before call to waitForTransform,
    // to avoid a tf2_ros::CreateTimerInterfaceException exception
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(this->get_node_base_interface(), this->get_node_timers_interface());
    tf2_buffer->setCreateTimerInterface(timer_interface);
    tf2_listener = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer);

    std::string shared_dir = ament_index_cpp::get_package_share_directory("enemy_predictor");
    RCLCPP_INFO(get_logger(), "shared_dir: %s", shared_dir.c_str());

    bac = std::make_shared<ballistic>(this, shared_dir);
    bac->refresh_velocity(false, 30.);  // 初始化为30

    enemy_armor_type.fill(0);
    enemy_armor_type[armor_type::HERO] = 1;
    armor_type_filter = std::vector<int>(20, 0);
    // off_cmd
    auto_shoot_from_pc_t shoot_behavior(0, 0, 0, 0, 15);
    off_cmd = make_cmd(shoot_behavior);
    detection_sub = this->create_subscription<rm_interfaces::msg::Detection>(
        params.detection_name, rclcpp::SensorDataQoS(), std::bind(&EnemyPredictorNode::detection_callback, this, std::placeholders::_1));

    robot_sub = this->create_subscription<rm_interfaces::msg::Rmrobot>(params.robot_name, rclcpp::SensorDataQoS(),
                                                                       std::bind(&EnemyPredictorNode::robot_callback, this, std::placeholders::_1));
    show_enemies_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("show_enemies", rclcpp::QoS(10).reliable().durability_volatile());
    pnp_pose_pub = this->create_publisher<nav_msgs::msg::Odometry>("show_pnp", rclcpp::QoS(10).reliable().durability_volatile());

    control_pub = this->create_publisher<rm_interfaces::msg::Control>(params.robot_name + "_control", rclcpp::SensorDataQoS());

    for (int i = 0; i < 10; ++i) {
        std::stringstream index;
        index << (char)('0' + i);
        watch_data_pubs.push_back(
            this->create_publisher<std_msgs::msg::Float64>(params.robot_name + "_EnemyPredictor_watchdata" + index.str(), rclcpp::SensorDataQoS()));
    }

    for (int i = 0; i < 4; ++i) {
        std::stringstream index;
        index << (char)('0' + i);
        armor_disyaw_llimit_pubs.push_back(this->create_publisher<geometry_msgs::msg::PoseStamped>("disyaw_llimit" + index.str(), rclcpp::QoS(10).reliable().durability_volatile()));
        armor_disyaw_rlimit_pubs.push_back(this->create_publisher<geometry_msgs::msg::PoseStamped>("disyaw_rlimit" + index.str(), rclcpp::QoS(10).reliable().durability_volatile()));
        armor_yaw_pubs.push_back(
            this->create_publisher<nav_msgs::msg::Odometry>("armor_yaw" + index.str(), rclcpp::QoS(10).reliable().durability_volatile()));
    }
}

EnemyPredictorNode::~EnemyPredictorNode() {}

void EnemyPredictorNode::load_params() {
    params.detection_name = this->declare_parameter("detection_name", "114514_detection");
    params.robot_name = this->declare_parameter("robot_name", "114514_robot");
    params.enable_imshow = this->declare_parameter("enable_imshow", false);
    params.debug = this->declare_parameter("debug", false);
    params.camera_name = this->declare_parameter("camera_name", "114514_camera");
    params.target_frame = this->declare_parameter("target_frame", "114514_frame");

    RCLCPP_INFO(get_logger(), "detection_name: %s", params.detection_name.c_str());
    RCLCPP_INFO(get_logger(), "robot_name: %s", params.robot_name.c_str());
    RCLCPP_INFO(get_logger(), "enable_imshow: %s", params.enable_imshow ? "True" : "False");
    RCLCPP_INFO(get_logger(), "debug: %s", params.debug ? "True" : "False");
    RCLCPP_INFO(get_logger(), "camera_name: %s", params.camera_name.c_str());
    RCLCPP_INFO(get_logger(), "target_frame: %s", params.target_frame.c_str());
    bool found_params = declare_parameter("found_params", false);
    assert(found_params && "EnemyPredictor:No valid Parameters!");
    // armor_ekf
    std::vector<double> vec_p = declare_parameter("armor_ekf.P", std::vector<double>());
    std::vector<double> vec_Q = declare_parameter("armor_ekf.Q", std::vector<double>());
    std::vector<double> vec_R = declare_parameter("armor_ekf.R", std::vector<double>());
    std::vector<double> vec_Ke = declare_parameter("armor_ekf.Ke", std::vector<double>());
    assert(vec_p.size() == 6 && "armor_ekf.P must be of size 6!");
    assert(vec_Q.size() == 6 && "armor_ekf.Q must be of size 6!");
    assert(vec_R.size() == 3 && "armor_ekf.R must be of size 3!");
    assert(vec_Ke.size() == 3 && "armor_ekf.Ke must be of size !");
    params.armor_ekf_config.P = armor_EKF::Vx(vec_p.data());
    params.armor_ekf_config.Q = armor_EKF::Vx(vec_Q.data());
    params.armor_ekf_config.R = armor_EKF::Vy(vec_R.data());
    params.armor_ekf_config.Ke = armor_EKF::Vy(vec_Ke.data());
    params.armor_ekf_config.length = declare_parameter("armor_ekf.filter_length", -1);
    params.armor_ekf_config.const_dis = declare_parameter("armor_ekf.const_dis", 0.0);

    // yaw_kf
    std::vector<double> vec_p_yaw = declare_parameter("yaw_kf.P", std::vector<double>());
    params.yaw_kf_config.sigma2_Q = declare_parameter("yaw_kf.Q", 1.0);
    std::vector<double> vec_R_yaw = declare_parameter("yaw_kf.R", std::vector<double>());
    assert(vec_p_yaw.size() == 2 && "armor_ekf.P must be of size 2!");
    assert(vec_R_yaw.size() == 1 && "armor_ekf.R must be of size 1!");
    params.yaw_kf_config.P = yaw_KF::Vx(vec_p_yaw.data());
    params.yaw_kf_config.R = yaw_KF::Vy(vec_R_yaw.data());

    // // enemy_ekf(自适应R/Q)
    // vec_p = declare_parameter("enemy_ekf.P", std::vector<double>());
    // assert(vec_p.size() == 13 && "armor_ekf.P must be of size 13!");
    // params.enemy_ekf_config.P = enemy_double_observer_EKF::Vn(vec_p.data());
    // params.enemy_ekf_config.R_XYZ = declare_parameter("enemy_ekf.R_XYZ", 0.0);
    // params.enemy_ekf_config.R_YAW = declare_parameter("enemy_ekf.R_YAW", 0.0);
    // params.enemy_ekf_config.R_PY = declare_parameter("enemy_ekf.R_PY", 0.0);
    // params.enemy_ekf_config.R_D = declare_parameter("enemy_ekf.R_D", 0.0);
    // params.enemy_ekf_config.R_R = declare_parameter("enemy_ekf.R_R", 0.0);
    // params.enemy_ekf_config.Q2_XYZ = declare_parameter("enemy_ekf.Q2_XYZ", 0.0);
    // params.enemy_ekf_config.Q2_YAW = declare_parameter("enemy_ekf.Q2_YAW", 0.0);
    // params.enemy_ekf_config.Q2_R = declare_parameter("enemy_ekf.Q2_R", 0.0);
    // params.enemy_ekf_config.predict_compensate = declare_parameter("predict_compensate", 1.1);

    armor_EKF::init(params.armor_ekf_config);
    yaw_KF::init(params.yaw_kf_config);
    // enemy_double_observer_EKF::init(params.enemy_ekf_config);

    // 传统方法感知陀螺/前哨战相关参数
    params.census_period_min = declare_parameter("census_period_min", 0.0);
    params.census_period_max = declare_parameter("census_period_max", 0.0);
    params.anti_outpost_census_period = declare_parameter("anti_outpost_census_period", 0.0);
    params.anti_outpost_census_period_min = declare_parameter("anti_outpost_census_period_min", 0.0);
    params.timestamp_thresh = declare_parameter("timestamp_thresh", 0.0);
    params.top_pitch_thresh = declare_parameter("top_pitch_thresh", 0.0);
    params.outpost_top_offset_dis = declare_parameter("outpost_top_offset_dis", 0.0);
    params.outpost_top_offset_z = declare_parameter("outpost_top_offset_z", 0.0);
    // 装甲目标过滤/选择
    params.sight_limit = declare_parameter("sight_limit", 0.0);
    params.high_limit = declare_parameter("high_limit", 0.0);
    params.size_limit = declare_parameter("size_limit", 0.0);
    params.bound_limit = declare_parameter("bound_limit", 0.0);
    params.aspect_limit_big = declare_parameter("aspect_limit_big", 0.0);
    params.aspect_limit_small = declare_parameter("aspect_limit_small", 0.0);
    params.rm_pnp_aspect_limit_big = declare_parameter("rm_pnp_aspect_limit_big", 0.0);
    params.rm_pnp_aspect_limit_small = declare_parameter("rm_pnp_aspect_limit_small", 0.0);
    params.reset_time = declare_parameter("reset_time", 0.0);
    params.size_ratio_thresh = declare_parameter("size_ratio_thresh", 0.0);
    std::vector<double> collimation_vec = declare_parameter("collimation", std::vector<double>());
    assert(collimation_vec.size() == 2 && "collimation size must be 2!");
    params.collimation.x = collimation_vec[0];
    params.collimation.y = collimation_vec[1];

    // 帧间匹配
    params.interframe_dis_thresh = declare_parameter("interframe_dis_thresh", 0.0);
    params.id_inertia = declare_parameter("id_inertia", 0);
    params.robot_2armor_dis_thresh = declare_parameter("robot_2armor_dis_thresh", 0.0);
    // 运动状态判断
    params.rotate_thresh = declare_parameter("rotate_thresh", 0.0);
    params.rotate_exit = declare_parameter("rotate_exit", 0.0);
    params.high_spd_rotate_thresh = declare_parameter("high_spd_rotate_thresh", 0.0);
    params.high_spd_rotate_exit = declare_parameter("high_spd_rotate_exit", 0.0);
    params.move_thresh = declare_parameter("move_thresh", 0.0);
    params.move_exit = declare_parameter("move_exit", 0.0);
    // 火控参数
    params.change_armor_time_thresh = declare_parameter("change_armor_time_thresh", 0.0);
    params.dis_thresh_kill = declare_parameter("dis_thresh_kill", 0.0);
    params.low_spd_thresh = declare_parameter("low_spd_thresh", 0.0);
    params.gimbal_error_dis_thresh = declare_parameter("gimbal_error_dis_thresh", 0.0);
    params.gimbal_error_dis_thresh_old = declare_parameter("gimbal_error_dis_thresh_old", 0.0);
    params.residual_thresh = declare_parameter("residual_thresh", 0.0);
    params.tangential_spd_thresh = declare_parameter("tangential_spd_thresh", 0.0);
    params.normal_spd_thresh = declare_parameter("normal_spd_thresh", 0.0);
    params.decel_delay_time = declare_parameter("decel_delay_time", 0.0);
    params.choose_enemy_without_autoaim_signal = declare_parameter("choose_enemy_without_autoaim_signal", false);
    params.disable_auto_shoot = declare_parameter("disable_auto_shoot", false);
    // 延迟参数
    params.response_delay = declare_parameter("response_delay", 0.0);
    params.shoot_delay = declare_parameter("shoot_delay", 0.0);

    params.rmcv_id = UNKNOWN_ID;
}

void EnemyPredictorNode::get_params() {
    this->get_parameter("enable_imshow", params.enable_imshow);
    this->get_parameter("debug", params.debug);
    // armor_ekf 参数
    std::vector<double> vec_p, vec_Q, vec_R, vec_Ke;
    this->get_parameter("armor_ekf.P", vec_p);
    this->get_parameter("armor_ekf.Q", vec_Q);
    this->get_parameter("armor_ekf.R", vec_R);
    this->get_parameter("armor_ekf.Ke", vec_Ke);
    params.armor_ekf_config.P = armor_EKF::Vx(vec_p.data());
    params.armor_ekf_config.Q = armor_EKF::Vx(vec_Q.data());
    params.armor_ekf_config.R = armor_EKF::Vy(vec_R.data());
    params.armor_ekf_config.Ke = armor_EKF::Vy(vec_Ke.data());
    this->get_parameter("armor_ekf.filter_length", params.armor_ekf_config.length);

    // enemy_ekf 参数
    // this->get_parameter("enemy_ekf.P", vec_p);
    // params.enemy_ekf_config.P = enemy_double_observer_EKF::Vn(vec_p.data());
    // this->get_parameter("enemy_ekf.R_XYZ", params.enemy_ekf_config.R_XYZ);
    // this->get_parameter("enemy_ekf.R_YAW", params.enemy_ekf_config.R_YAW);
    // this->get_parameter("enemy_ekf.R_PY", params.enemy_ekf_config.R_PY);
    // this->get_parameter("enemy_ekf.R_D", params.enemy_ekf_config.R_D);
    // this->get_parameter("enemy_ekf.R_R", params.enemy_ekf_config.R_R);
    // this->get_parameter("enemy_ekf.Q2_XYZ", params.enemy_ekf_config.Q2_XYZ);
    // this->get_parameter("enemy_ekf.Q2_YAW", params.enemy_ekf_config.Q2_YAW);
    // this->get_parameter("enemy_ekf.Q2_R", params.enemy_ekf_config.Q2_R);
    // this->get_parameter("predict_compensate", params.enemy_ekf_config.predict_compensate);

    // 与前哨站相关的参数
    this->get_parameter("census_period_min", params.census_period_min);
    this->get_parameter("census_period_max", params.census_period_max);
    this->get_parameter("anti_outpost_census_period", params.anti_outpost_census_period);
    this->get_parameter("anti_outpost_census_period_min", params.anti_outpost_census_period_min);
    this->get_parameter("timestamp_thresh", params.timestamp_thresh);
    this->get_parameter("top_pitch_thresh", params.top_pitch_thresh);
    this->get_parameter("outpost_top_offset_dis", params.outpost_top_offset_dis);
    this->get_parameter("outpost_top_offset_z", params.outpost_top_offset_z);

    // 装甲目标过滤/选择参数
    this->get_parameter("sight_limit", params.sight_limit);
    this->get_parameter("high_limit", params.high_limit);
    this->get_parameter("size_limit", params.size_limit);
    this->get_parameter("bound_limit", params.bound_limit);
    this->get_parameter("aspect_limit_big", params.aspect_limit_big);
    this->get_parameter("aspect_limit_small", params.aspect_limit_small);
    this->get_parameter("rm_pnp_aspect_limit_big", params.rm_pnp_aspect_limit_big);
    this->get_parameter("rm_pnp_aspect_limit_small", params.rm_pnp_aspect_limit_small);
    this->get_parameter("reset_time", params.reset_time);
    this->get_parameter("size_ratio_thresh", params.size_ratio_thresh);
    std::vector<double> collimation_vec;
    this->get_parameter("collimation", collimation_vec);
    params.collimation.x = collimation_vec[0];
    params.collimation.y = collimation_vec[1];

    // 帧间匹配参数
    this->get_parameter("interframe_dis_thresh", params.interframe_dis_thresh);
    this->get_parameter("id_inertia", params.id_inertia);
    this->get_parameter("robot_2armor_dis_thresh", params.robot_2armor_dis_thresh);

    // 运动状态判断参数
    this->get_parameter("rotate_thresh", params.rotate_thresh);
    this->get_parameter("rotate_exit", params.rotate_exit);
    this->get_parameter("high_spd_rotate_thresh", params.high_spd_rotate_thresh);
    this->get_parameter("high_spd_rotate_exit", params.high_spd_rotate_exit);
    this->get_parameter("move_thresh", params.move_thresh);
    this->get_parameter("move_exit", params.move_exit);

    // 火控参数
    this->get_parameter("dis_thresh_kill", params.dis_thresh_kill);
    this->get_parameter("low_spd_thresh", params.low_spd_thresh);
    this->get_parameter("gimbal_error_dis_thresh", params.gimbal_error_dis_thresh);
    this->get_parameter("gimbal_error_dis_thresh_old", params.gimbal_error_dis_thresh_old);
    this->get_parameter("residual_thresh", params.residual_thresh);
    this->get_parameter("tangential_spd_thresh", params.tangential_spd_thresh);
    this->get_parameter("normal_spd_thresh", params.normal_spd_thresh);
    this->get_parameter("decel_delay_time", params.decel_delay_time);
    this->get_parameter("choose_enemy_without_autoaim_signal", params.choose_enemy_without_autoaim_signal);
    this->get_parameter("disable_auto_shoot", params.disable_auto_shoot);

    // 延迟参数
    this->get_parameter("response_delay", params.response_delay);
    this->get_parameter("shoot_delay", params.shoot_delay);
}

void EnemyPredictorNode::add_point_Marker(double x_, double y_, double z_, double r_, double g_, double b_, double a_, Eigen::Vector3d pos) {
    visualization_msgs::msg::MarkerArray marker_array;
    visualization_msgs::msg::Marker marker;
    // 画中心
    marker.header.frame_id = "odom";
    marker.header.stamp = rclcpp::Node::now();
    marker.ns = "points";
    marker.id = marker_id++;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = pos[0];
    marker.pose.position.y = pos[1];
    marker.pose.position.z = pos[2];
    marker.pose.orientation.w = 1.0;
    marker.scale.x = x_;  // 球的大小
    marker.scale.y = y_;
    marker.scale.z = z_;
    marker.color.r = r_;  // 球的颜色
    marker.color.g = g_;
    marker.color.b = b_;
    marker.color.a = a_;
    markers.markers.push_back(marker);
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<enemy_predictor::EnemyPredictorNode>(rclcpp::NodeOptions()));
    if (rclcpp::ok()) {
        rclcpp::shutdown();
    }
    return 0;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(enemy_predictor::EnemyPredictorNode)