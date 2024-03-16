#include <rm_serial_driver/rm_serial_driver.h>
#include <rm_utils/data.h>
#include <rm_utils/datatypes.h>
#include <rm_utils/soft_crc.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>

using namespace rm_serial_driver;

RMSerialDriver::RMSerialDriver(const rclcpp::NodeOptions& options)
    : Node("rm_serial_driver", options) {
    RCLCPP_INFO(get_logger(), "Start RMSerialDriver!");

    getParams();

    // TF broadcaster
    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Robot publisher
    robot_pub = create_publisher<rm_interfaces::msg::Rmrobot>(robot_name, rclcpp::SensorDataQoS());

    joint_state_pub =
        create_publisher<sensor_msgs::msg::JointState>("joint_states", rclcpp::SensorDataQoS());

    // Control subscription
    control_sub = create_subscription<rm_interfaces::msg::Control>(
        robot_name + "_control", rclcpp::SensorDataQoS(),
        std::bind(&RMSerialDriver::ControlMsgCallback, this, std::placeholders::_1));

    try {
        serial_driver = std::make_unique<serial::Serial>(device_name, baud_rate,
                                                         serial::Timeout::simpleTimeout(1000));
        if (!serial_driver->isOpen()) {
            serial_driver->open();
        }
        receive_thread = std::thread(&RMSerialDriver::receiveData, this);
    } catch (const std::exception& ex) {
        RCLCPP_ERROR(get_logger(), "Error creating serial port: %s - %s", device_name.c_str(),
                     ex.what());
        throw ex;
    }
}

RMSerialDriver::~RMSerialDriver() {
    if (receive_thread.joinable()) {
        receive_thread.join();
    }

    if (serial_driver->isOpen()) {
        serial_driver->close();
    }
}

void RMSerialDriver::imuMsgCallback(recv_msg* msg) {
    // RCLCPP_INFO(get_logger(), "recv! %lf %lf %lf", msg->roll, msg->pitch, msg->yaw);
    static long long last_time = -1;
    static long long now_time;
    now_time = this->now().nanoseconds();
    if (last_time != -1) {
        long long diff_time = now_time - last_time;
        if (diff_time > 5000000) {
            RCLCPP_INFO(get_logger(), "too big diff time!: %lf ms", diff_time / (double)1000000);
        }
    }
    last_time = now_time;

    // tf2::Quaternion pitch2yaw_r, yaw2odom_r;
    // pitch2yaw_r.setRPY(msg->roll, msg->pitch, 0);
    // yaw2odom_r.setRPY(0, 0, msg->yaw);
    // tf2::Transform pitch2yaw(pitch2yaw_r,
    //  tf2::Vector3(pitch2yaw_t[0], pitch2yaw_t[1], pitch2yaw_t[2]));
    // tf2::Transform yaw2odom(yaw2odom_r);
    // tf2::Transform gimbal2odom = yaw2odom * pitch2yaw;
    //
    // geometry_msgs::msg::TransformStamped t;
    // timestamp_offset = this->get_parameter("timestamp_offset").as_double();
    // t.header.stamp = this->now() + rclcpp::Duration::from_seconds(timestamp_offset);
    // t.header.frame_id = "odom";
    // t.child_frame_id = "gimbal_link";
    // t.transform = tf2::toMsg(gimbal2odom);
    // tf_broadcaster->sendTransform(t);

    sensor_msgs::msg::JointState joint_msg;
    joint_msg.header.stamp = this->now() + rclcpp::Duration::from_seconds(timestamp_offset);
    std::vector<std::string> names = {"yaw_joint", "pitch_joint"};
    std::vector<double> positions = {msg->yaw, msg->pitch};
    joint_msg.name = names;
    joint_msg.position = positions;
    joint_state_pub->publish(joint_msg);

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
    // robot_msg.robot_id = 5;
    robot_pub->publish(robot_msg);
}

void RMSerialDriver::robotpub_low_freq(const rm_interfaces::msg::Rmrobot& msg) {
    static int count_down = robotpub_count_down;
    count_down--;
    if (count_down == 0) {
        count_down = robotpub_count_down;
        robot_pub->publish(msg);
    }
}

void RMSerialDriver::reopenPort() {
    RCLCPP_WARN(get_logger(), "Attempting to reopen port");
    try {
        if (serial_driver->isOpen()) {
            serial_driver->close();
        }
        serial_driver->open();
        RCLCPP_INFO(get_logger(), "Successfully reopened port");
    } catch (const std::exception& ex) {
        RCLCPP_ERROR(get_logger(), "Error while reopening port: %s", ex.what());
        if (rclcpp::ok()) {
            rclcpp::sleep_for(std::chrono::seconds(1));
            reopenPort();
        }
    }
}

void RMSerialDriver::receiveData() {
    std::vector<uint8_t> header(2);
    std::vector<uint8_t> buffer_vec;
    int data_len = sizeof(recv_msg);
    buffer_vec.resize(data_len + 5);
    uint8_t* buffer = buffer_vec.data();
    recv_msg* rmsg = (recv_msg*)(buffer + 2);
    int recv_state = 0;
    int no_serial_data = 0;
    // rclcpp::sleep_for(std::chrono::seconds(1));
    while (rclcpp::ok()) {
        try {
            if (recv_state == 0) {
                int res = serial_driver->read(buffer, 2);
                if (res != 2) {
                    ++no_serial_data;
                } else if (buffer[0] == 's' && buffer[1] == data_len) {
                    recv_state = 1;
                } else {
                    // RCLCPP_WARN(get_logger(), "Data Header Check Failed!");
                }
            } else if (recv_state == 1) {
                int res = serial_driver->read(buffer + 2, data_len + 3);
                if (res != data_len + 3) {
                    ++no_serial_data;
                } else if (buffer[data_len + 4] == 'e' &&
                           buffer_check_valid(buffer + 1, data_len + 3, CRC16::crc16_ccitt)) {
                    imuMsgCallback(rmsg);
                    no_serial_data = 0;
                } else {
                    RCLCPP_WARN(get_logger(), "Data CRC Check Failed!");
                }
                recv_state = 0;
            }

            if (no_serial_data > 5) {
                RCLCPP_WARN(get_logger(), "no serial data....");
                no_serial_data = 0;
            }

        } catch (const std::exception& ex) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 20, "Error while receiving data: %s",
                                  ex.what());
            reopenPort();
        }
    }
}

void RMSerialDriver::getParams() {
    timestamp_offset = this->declare_parameter("timestamp_offset", 0.0);
    robot_name = this->declare_parameter("robot_name", "114514_robot");
    pitch2yaw_t = this->declare_parameter("pitch2yaw_t", std::vector<double>({0, 0, 0}));
    if (pitch2yaw_t.size() != 3) {
        throw std::invalid_argument{"The pitch2yaw_t parameter must be of size 3."};
    } else {
        RCLCPP_INFO(get_logger(), "pitch2yaw_t: %lf %lf %lf", pitch2yaw_t[0], pitch2yaw_t[1],
                    pitch2yaw_t[2]);
    }
    RCLCPP_INFO(get_logger(), "Receive IMU ID: %x", imu_msg_id);

    try {
        device_name = declare_parameter<std::string>("device_name", "");
    } catch (rclcpp::ParameterTypeException& ex) {
        RCLCPP_ERROR(get_logger(), "The device name provided was invalid");
        throw ex;
    }

    try {
        baud_rate = declare_parameter<int>("baud_rate", 0);
        RCLCPP_INFO(get_logger(), "baud_rate: %d", baud_rate);
    } catch (rclcpp::ParameterTypeException& ex) {
        RCLCPP_ERROR(get_logger(), "The baud_rate provided was invalid");
        throw ex;
    }
}

void RMSerialDriver::ControlMsgCallback(rm_interfaces::msg::Control::ConstSharedPtr control_msg) {
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
    // msg->vitual_mode = control_msg->flag;
    msg->rate = control_msg->rate;
    msg->one_shot_num = control_msg->one_shot_num;
    msg->vision_follow_id = control_msg->vision_follow_id;
    *crc_now = CRC16::crc16_ccitt.check_sum(send_buffer + 1, data_len + 1);
    send_buffer[buffer_len - 1] = 'e';

    try {
        serial_driver->write(send_buffer_vec);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Send Failed: %s", e.what());
    }
}

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_serial_driver::RMSerialDriver)
