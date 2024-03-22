#include <rm_can_driver/rm_can_driver.h>

#include <rm_can_driver/socket_can_id.hpp>
using namespace rm_can_driver;

RMCanDriver::RMCanDriver(const rclcpp::NodeOptions& options) : Node("rm_can_driver", options) {
    RCLCPP_INFO(get_logger(), "Staring Robomaster can driver!");
    loadParams();
    filter_list.filters.push_back({(uint32_t)recv_id, CAN_SFF_MASK});
    reopenPort();
    rm_receiver = std::make_unique<msg_receiver<recv_msg>>(
        std::bind(&RMCanDriver::imuMsgCallback, this, std::placeholders::_1), this->get_logger());

    // TF broadcaster
    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Robot publisher
    robot_pub = create_publisher<rm_interfaces::msg::Rmrobot>(robot_name, rclcpp::SensorDataQoS()); 
    // Control subscription
    control_sub = create_subscription<rm_interfaces::msg::Control>(
        robot_name + "_control", rclcpp::SensorDataQoS(),
        std::bind(&RMCanDriver::ControlMsgCallback, this, std::placeholders::_1));
    receive_thread = std::thread(&RMCanDriver::receiveData, this);
}

void RMCanDriver::loadParams() {
    timestamp_offset = this->declare_parameter("timestamp_offset", 0.0);
    robot_name = this->declare_parameter("robot_name", "114514_robot");
    sudo_pwd = this->declare_parameter("sudo_pwd", "1234567");
    device_name = this->declare_parameter("device_name", "can114514");
    recv_id = this->declare_parameter("recv_id", 0x15);
    send_id = this->declare_parameter("send_id", 0x234);
    baud_rate = this->declare_parameter("baud_rate", 0);
    pitch2yaw_t = this->declare_parameter("pitch2yaw_t", std::vector<double>({0, 0, 0}));
    if (pitch2yaw_t.size() != 3) {
        throw std::invalid_argument{"The pitch2yaw_t parameter must be of size 3."};
    } else {
        RCLCPP_INFO(get_logger(), "pitch2yaw_t: %lf %lf %lf", pitch2yaw_t[0], pitch2yaw_t[1],
                    pitch2yaw_t[2]);
    }
    RCLCPP_INFO(get_logger(), "Can Device: %s", device_name.c_str());
    RCLCPP_INFO(get_logger(), "Receive Can ID: %x", recv_id);
    RCLCPP_INFO(get_logger(), "Send Can ID: %x", send_id);
}

void RMCanDriver::reopenPort() {
    if (can_receiver) {
        can_receiver.reset();
        can_sender.reset();
    }

    std::string sys_cmd = "echo " + sudo_pwd + "|sudo -S ifconfig " + device_name + " down";
    int res = system(sys_cmd.c_str());
    RCLCPP_INFO(get_logger(), "system: %s res: %d", sys_cmd.c_str(), res);

    sys_cmd = "echo " + sudo_pwd + "|sudo -S ip link set " + device_name + " type can bitrate " +
              std::to_string(baud_rate);
    res = system(sys_cmd.c_str());
    RCLCPP_INFO(get_logger(), "system: %s res: %d", sys_cmd.c_str(), res);

    sys_cmd = "echo " + sudo_pwd + "|sudo -S ifconfig " + device_name + " up";
    res = system(sys_cmd.c_str());
    RCLCPP_INFO(get_logger(), "system: %s res: %d", sys_cmd.c_str(), res);

    sys_cmd = "echo " + sudo_pwd + "|sudo -S ifconfig " + device_name + " txqueuelen 1000";
    res = system(sys_cmd.c_str());
    RCLCPP_INFO(get_logger(), "system: %s res: %d", sys_cmd.c_str(), res);

    // try to reinit can device
    bool create_success = false;
    while (!create_success && rclcpp::ok()) {
        try {
            can_receiver =
                std::make_unique<drivers::socketcan::SocketCanReceiver>(device_name, false);
            can_sender = std::make_unique<drivers::socketcan::SocketCanSender>(device_name, false);
            can_receiver->SetCanFilters(filter_list);
            create_success = true;
        } catch (const std::exception& ex) {
            RCLCPP_ERROR(get_logger(), "Error when creating can interface: %s", ex.what());
            RCLCPP_ERROR(get_logger(), "Restarting...");
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
}

void RMCanDriver::receiveData() {
    std::vector<uint8_t> buffer;
    buffer.resize(20);
    int fail_cnt = 0;
    while (rclcpp::ok()) {
        try {
            auto id = can_receiver->receive(buffer.data(), std::chrono::milliseconds(20));
            rm_receiver->receive(buffer.data(), id.length());
            fail_cnt = 0;
        } catch (const std::exception& ex) {
            RCLCPP_WARN(get_logger(), "Error while receiving data: %s", ex.what());
            fail_cnt++;
            if (fail_cnt == 10) {
                fail_cnt = 0;
                RCLCPP_WARN(get_logger(), "Trying to reopen port....");
                std::this_thread::sleep_for(std::chrono::seconds(2));
                reopenPort();
            }
        }
    }
}

void RMCanDriver::imuMsgCallback(recv_msg* msg) {
    // RCLCPP_INFO(get_logger(), "recv! %lf %lf %lf", msg->roll, msg->pitch, msg->yaw);
    tf2::Quaternion pitch2yaw_r, yaw2odom_r;
    pitch2yaw_r.setRPY(msg->roll, msg->pitch, 0);
    yaw2odom_r.setRPY(0, 0, msg->yaw);
    tf2::Transform pitch2yaw(pitch2yaw_r,
                             tf2::Vector3(pitch2yaw_t[0], pitch2yaw_t[1], pitch2yaw_t[2]));
    tf2::Transform yaw2odom(yaw2odom_r);
    tf2::Transform gimbal2odom = yaw2odom * pitch2yaw;

    geometry_msgs::msg::TransformStamped t;
    timestamp_offset = this->get_parameter("timestamp_offset").as_double();
    t.header.stamp = this->now() + rclcpp::Duration::from_seconds(timestamp_offset);
    t.header.frame_id = "odom";
    t.child_frame_id = "gimbal_link";
    t.transform = tf2::toMsg(gimbal2odom);
    tf_broadcaster->sendTransform(t);

    rm_interfaces::msg::Rmrobot robot_msg;
    robot_msg.robot_id = msg->robot_id;
    robot_msg.bullet_velocity = msg->bullet_speed;
    robot_msg.vision_mode = mode2string(cast_run_mode(msg->mode));
    robot_msg.right_press = msg->mode == 1;
    robot_msg.lobshot = msg->mode == 4;
    robot_msg.imu.roll = msg->roll;
    robot_msg.imu.pitch = msg->pitch;
    robot_msg.imu.yaw = msg->yaw;
    // robotpub_low_freq(robot_msg);
    robot_pub->publish(robot_msg);
}

void RMCanDriver::ControlMsgCallback(rm_interfaces::msg::Control::ConstSharedPtr control_msg) {
    static std::vector<uint8_t> send_buffer_vec;
    static int data_len = (int)sizeof(send_msg);
    // s/e 2 len 1 crc 2
    static int buffer_len = data_len + 5;
    send_buffer_vec.resize(buffer_len);
    uint8_t* send_buffer = send_buffer_vec.data();
    send_msg* msg = (send_msg*)(send_buffer + 2);
    uint16_t* crc_now = (uint16_t*)(send_buffer + 2 + data_len);

    send_buffer[0] = 's';
    send_buffer[1] = (uint8_t)data_len;
    msg->pitch = control_msg->pitch;
    msg->yaw = control_msg->yaw;
    msg->flag = control_msg->flag;
    msg->rate = control_msg->rate;
    msg->one_shot_num = control_msg->one_shot_num;
    msg->vision_follow_id = control_msg->vision_follow_id;
    *crc_now = CRC16::crc16_ccitt.check_sum(send_buffer + 1, data_len + 1);
    send_buffer[buffer_len - 1] = 'e';

    for (uint32_t idx = 0; idx < (uint32_t)buffer_len; idx += 8) {
        uint32_t dlc = buffer_len - idx >= 8 ? 8 : buffer_len - idx;
        drivers::socketcan::CanId send_conf(send_id, 0, drivers::socketcan::FrameType::DATA,
                                            drivers::socketcan::StandardFrame);
        try {
            can_sender->send(send_buffer + idx, dlc, send_conf, std::chrono::milliseconds(20));
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Send Failed: %s", e.what());
        }
    }

    // send_buffer
    // auto can_id_conf = drivers::socketcan::CanId(send_id,0,)
}

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_can_driver::RMCanDriver)