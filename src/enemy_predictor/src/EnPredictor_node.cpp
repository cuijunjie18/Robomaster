#include <enemy_predictor/ekf.h>
#include <enemy_predictor/enemy_predictor.h>

#include <Eigen/Core>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <functional>
using namespace enemy_predictor;
std::vector<cv::Vec3d> EnemyPredictorNode::SmallArmor = {
    // 单位：m
    {-0.0675, 0.0275, 0.},
    {-0.0675, -0.0275, 0.},
    {0.0675, -0.0275, 0.},
    {0.0675, 0.0275, 0.},
};
std::vector<cv::Vec3d> EnemyPredictorNode::BigArmor = {
    // 单位：m
    {-0.115, 0.029, 0.},
    {-0.115, -0.029, 0.},
    {0.115, -0.029, 0.},
    {0.115, 0.029, 0.},
};
std::vector<cv::Vec3d> EnemyPredictorNode::pw_energy = {  // 单位：m
    {-0.1542, -0.15456, 0.},
    {0.1542, -0.15456, 0.},
    {0.18495, 0.15839, 0.},
    {0., 0.52879, 0.},
    {-0.18495, 0.15839, 0.}};
std::vector<cv::Vec3d> EnemyPredictorNode::pw_result = {  // 单位：m
    {-0.18495, 0.15839, 0.},
    {-0.1542, -0.15456, 0.},
    {0.1542, -0.15456, 0.},
    {0.18495, 0.15839, 0.},
    {0., 0.7, 0.}};

/** \brief 给定起点和终点的frame_id，计算坐标变换
 * \param target_frame 起点的frame_id
 * \param source_frame 终点的frame_id
 * \param source_point 待转换的坐标
 * \return 转换后的坐标
 */
Eigen::Vector3d EnemyPredictorNode::trans(const std::string &target_frame, const std::string &source_frame, Eigen::Vector3d source_point) {
    Eigen::Vector3d result;
    geometry_msgs::msg::TransformStamped t;
    try {
        t = tf2_buffer->lookupTransform(target_frame, source_frame, detection_header.stamp, rclcpp::Duration::from_seconds(0.5));

    } catch (const std::exception &ex) {
        RCLCPP_ERROR(this->get_logger(), "Could not transform %s to %s: %s", source_frame, target_frame.c_str(), ex.what());
        abort();
    }
    tf2::doTransform<Eigen::Vector3d>(source_point, result, t);
    return result;
}

cv::Point2d EnemyPredictorNode::pos2img(Eigen::Matrix<double, 3, 1> X) {
    X = trans(detection_header.frame_id, "odom", X);
    X = K * X / X[2];
    return cv::Point2d(X[0], X[1]);
}

EnemyPredictorNode::EnemyPredictorNode(const rclcpp::NodeOptions &options) : Node("enemy_predictor", options) {
    RCLCPP_INFO(get_logger(), "EnemyPredictor Start!");
    load_params();

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
    // off_cmd
    off_cmd = make_cmd(0, 0, 0, 0, 15);
    detection_sub = this->create_subscription<rm_interfaces::msg::Detection>(
        params.detection_name, rclcpp::SensorDataQoS(), std::bind(&EnemyPredictorNode::detection_callback, this, std::placeholders::_1));

    robot_sub = this->create_subscription<rm_interfaces::msg::Rmrobot>(params.robot_name, rclcpp::SensorDataQoS(),
                                                                       std::bind(&EnemyPredictorNode::robot_callback, this, std::placeholders::_1));
    control_pub = this->create_publisher<rm_interfaces::msg::Control>(params.robot_name + "_control", rclcpp::SensorDataQoS());
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

    // enemy_ekf(自适应R/Q)
    vec_p = declare_parameter("enemy_ekf.P", std::vector<double>());
    assert(vec_p.size() == 9 && "armor_ekf.P must be of size 9!");
    params.enemy_ekf_config.P = enemy_half_observer_EKF::Vn(vec_p.data());
    params.enemy_ekf_config.R_XYZ = declare_parameter("enemy_ekf.R_XYZ", 0.0);
    params.enemy_ekf_config.R_YAW = declare_parameter("enemy_ekf.R_YAW", 0.0);
    params.enemy_ekf_config.Q2_XYZ = declare_parameter("enemy_ekf.Q2_XYZ", 0.0);
    params.enemy_ekf_config.Q2_YAW = declare_parameter("enemy_ekf.Q2_YAW", 0.0);
    params.enemy_ekf_config.Q2_R = declare_parameter("enemy_ekf.Q2_R", 0.0);

    armor_EKF::init(params.armor_ekf_config);
    enemy_half_observer_EKF::init(params.enemy_ekf_config);

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

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(enemy_predictor::EnemyPredictorNode)