#include <hik_camera/hik_camera.h>
#include <rm_utils/common.h>
#include <rm_utils/frame_info.h>
#include <unistd.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <rm_utils/perf.hpp>
#include <string>
#include <toml.hpp>
// 尝试func, 如果返回值不是MV_OK(即0)则调用logger记录WARN日志
#define UPDBW(func)                                                                        \
    nRet = func;                                                                           \
    if (nRet != MV_OK) {                                                                   \
        RCLCPP_WARN(this->get_logger(), #func " failed!, error code: %x", (unsigned)nRet); \
    }

// 尝试func, 如果返回值不是MV_OK(即0)则调用logger记录FATAL日志
#define UPDBF(func)                                                                         \
    nRet = func;                                                                            \
    if (nRet != MV_OK) {                                                                    \
        RCLCPP_FATAL(this->get_logger(), #func " failed!, error code: %x", (unsigned)nRet); \
    }

// 对于不可恢复性错误重启相机节点
#define UPDBE(func)      \
    UPDBF(func)          \
    if (nRet != MV_OK) { \
        reset();         \
    }

using namespace hik_camera;

HikCameraNode::HikCameraNode(const rclcpp::NodeOptions& options) : Node("hik_camera", options) {
    RCLCPP_INFO(this->get_logger(), "Starting HikCameraNode!");
    load_params();
    // 创建pub
    bool use_intra = options.use_intra_process_comms();
    if (!use_intra) {
        RCLCPP_WARN(get_logger(), "Not In Intra Process Mode");
    }
    if (!params.use_sensor_data_qos) {
        RCLCPP_WARN(get_logger(), "Not Use Sensor Data Qos");
    }
    auto qos = params.use_sensor_data_qos ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;
    image_pub = image_transport::create_publisher(this, params.cam_name, qos);

    // 创建sub
    robot_sub = create_subscription<rm_interfaces::msg::Rmrobot>(
        params.robot_name, rclcpp::SensorDataQoS(),
        std::bind(&HikCameraNode::robot_mode_update, this, std::placeholders::_1));
    // load camera info
    camera_info_manager =
        std::make_unique<camera_info_manager::CameraInfoManager>(this, params.cam_name);
    if (camera_info_manager->validateURL(params.camera_info_url)) {
        camera_info_manager->loadCameraInfo(params.camera_info_url);
        camera_info_msg = camera_info_manager->getCameraInfo();
    } else {
        RCLCPP_WARN(this->get_logger(), "Invalid camera info URL: %s",
                    params.camera_info_url.c_str());
    }
    init_camera();
    RCLCPP_WARN(get_logger(), "Starting Camera Monitor thread.");
    monitor_on = true;
    monitor_thread = std::thread(&HikCameraNode::monitor, this);
};

HikCameraNode::~HikCameraNode() {
    monitor_on = false;
    if (monitor_thread.joinable()) {
        monitor_thread.join();
    }
    close_device();
};

void HikCameraNode::load_params() {
    params.mode = AUTO_AIM;
    params.sn = this->declare_parameter("sn", "");
    params.cam_name = this->declare_parameter("camera_name", "camera_raw");
    params.robot_name = this->declare_parameter("robot_name", "robot114514");
    params.camera_info_url =
        this->declare_parameter("camera_info_url", "package://hik_camera/config/camera_info.yaml");
    // normal
    params.offset_x = this->declare_parameter("offset_x", -1);
    params.offset_y = this->declare_parameter("offset_y", -1);
    params.out_post_top_offset_y = this->declare_parameter("out_post_top_offset_y", -1);
    params.roi_height = this->declare_parameter("roi_height", 1024);
    params.roi_width = this->declare_parameter("roi_width", 1024);
    params.exposure_time = this->declare_parameter("exposure_time", 4000.0);
    params.gain = this->declare_parameter("gain", 15.0);
    params.gamma = this->declare_parameter("gamma", 0.5);
    params.digital_shift = this->declare_parameter("digital_shift", 6.0);
    // energy
    params.en_offset_x = this->declare_parameter("en_offset_x", -1);
    params.en_offset_y = this->declare_parameter("en_offset_y", -1);
    params.en_roi_height = this->declare_parameter("en_roi_height", 1024);
    params.en_roi_width = this->declare_parameter("en_roi_width", 1024);
    params.en_exposure_time = this->declare_parameter("en_exposure_time", 4000.0);
    params.en_gain = this->declare_parameter("en_gain", 15.0);
    params.en_resize_640 = this->declare_parameter("en_resize_640", true);

    params.frame_rate = this->declare_parameter("frame_rate", 100.0);
    params.use_sensor_data_qos = this->declare_parameter("use_sensor_data_qos", true);
}
void HikCameraNode::init_camera() {
    MV_CC_DEVICE_INFO_LIST device_list;
    RCLCPP_INFO(this->get_logger(), "Camera SN: %s", params.sn.c_str());
    bool device_found = false;
    while (!device_found && rclcpp::ok()) {
        // 枚举设备
        UPDBW(MV_CC_EnumDevices(MV_USB_DEVICE, &device_list));
        if (device_list.nDeviceNum > 0) {
            if (params.sn == "") {
                // 未设置camera sn,选择第一个
                RCLCPP_WARN(this->get_logger(), "Camera SN not set, use the first camera device");
                UPDBE(MV_CC_CreateHandle(&camera_handle, device_list.pDeviceInfo[0]));
                device_found = true;
            } else {
                static char device_sn[INFO_MAX_BUFFER_SIZE];
                for (size_t i = 0; i < device_list.nDeviceNum; ++i) {
                    memcpy(device_sn,
                           device_list.pDeviceInfo[i]->SpecialInfo.stUsb3VInfo.chSerialNumber,
                           INFO_MAX_BUFFER_SIZE);
                    device_sn[63] = '\0';  // 以防万一
                    RCLCPP_INFO(this->get_logger(), "Camera SN list: %s", device_sn);
                    if (std::strncmp(device_sn, params.sn.c_str(), 64U) == 0) {
                        UPDBE(MV_CC_CreateHandle(&camera_handle, device_list.pDeviceInfo[i]));
                        device_found = true;
                        break;
                    }
                }
            }
        }
        if (device_found) {
            break;
        } else {
            RCLCPP_WARN(this->get_logger(), "Camera SN:%s not found.", params.sn.c_str());
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    if (device_found) {
        open_device();
        set_hk_params();
        start_grab();
        Gamma_lookUpTable = cv::Mat(1, 256, CV_8U);
        uchar* p = Gamma_lookUpTable.ptr();
        for (int i = 0; i < 256; ++i) {
            p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, params.gamma) * 255.0);
        }
    }
}

void HikCameraNode::monitor() {
    while (rclcpp::ok() && monitor_on) {
        if (camera_failed) {
            RCLCPP_ERROR(this->get_logger(), "Camera failed! restarting...");
            reset();
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void HikCameraNode::start_grab() {
    // 开始采集
    MV_CC_StartGrabbing(camera_handle);
    // 开启采集线程
    grab_on = true;
    camera_failed = false;
    capture_thread = std::thread(&HikCameraNode::grab, this);
}

void HikCameraNode::stop_grab() {
    grab_on = false;
    if (capture_thread.joinable()) {
        capture_thread.join();
    }
    if (camera_handle) {
        MV_CC_StopGrabbing(camera_handle);
    }
}

std::pair<int, int> HikCameraNode::get_sensor_height_width() {
    // 获取max height/width
    MVCC_INTVALUE _max_height, _max_width;
    UPDBW(MV_CC_GetIntValue(camera_handle, "WidthMax", &_max_width));
    UPDBW(MV_CC_GetIntValue(camera_handle, "HeightMax", &_max_height));
    return std::pair{_max_height.nCurValue, _max_width.nCurValue};
}

void HikCameraNode::fit_int_step(std::string property, int& value) {
    if (value < 0) {
        throw std::invalid_argument("fit_in_step: value is negative, value: " +
                                    std::to_string(value));
    }
    MVCC_INTVALUE m;
    UPDBW(MV_CC_GetIntValue(camera_handle, property.c_str(), &m))
    int step = m.nInc;
    if (value % step) {
        RCLCPP_WARN(this->get_logger(),
                    "%s (current value: %d) does not meet the required incr(%d), auto swith to %d",
                    property.c_str(), value, step, value - value % step);
        value -= value % step;
    }
    if (value < (int)m.nMin || value > (int)m.nMax) {
        throw CameraException(
            "Property: '" + property + "'set value out of range, min: " + std::to_string(m.nMin) +
            ", max: " + std::to_string(m.nMax) + ", current: " + std::to_string(value));
    }
}

void HikCameraNode::set_grab_params(int offset_x, int offset_y, int roi_width, int roi_height) {
    // 先将offset置零的原因是, 这些SetValue会保持到下一次运行,
    // 如果原先存在的offset较大且与下一次设置的Width/Height加起来大于可用像素(如1080), SDK会拒绝设置
    // 先设置offset成正确值则存在如果事先设置的Width较大且本次设置的Offset也较大, 同样也会拒绝
    // 先置offset零然后设置就不会产生冲突
    UPDBW(MV_CC_SetIntValue(camera_handle, "OffsetX", 0))
    UPDBW(MV_CC_SetIntValue(camera_handle, "OffsetY", 0))
    UPDBW(MV_CC_SetIntValue(camera_handle, "Width", roi_width))
    UPDBW(MV_CC_SetIntValue(camera_handle, "Height", roi_height))

    fit_int_step("OffsetX", offset_x);
    fit_int_step("OffsetY", offset_y);

    UPDBW(MV_CC_SetIntValue(camera_handle, "OffsetX", offset_x))
    UPDBW(MV_CC_SetIntValue(camera_handle, "OffsetY", offset_y))
    // 获取有关Image的数据
    MV_CC_GetImageInfo(camera_handle, &img_info);
}

void HikCameraNode::set_hk_params() {
    const auto& [max_height, max_width] = get_sensor_height_width();
    // 自动调整offset
    if (params.offset_x == -1) params.offset_x = (max_width - params.roi_width) / 2;
    if (params.offset_y == -1) params.offset_y = (max_height - params.roi_height) / 2;
    RCLCPP_INFO(this->get_logger(), "roi_height: %d, roi_width: %d", params.roi_height,
                params.roi_width);
    RCLCPP_INFO(this->get_logger(), "offset_x: %d, offset_y: %d", params.offset_x, params.offset_y);

    if (params.en_offset_x == -1) params.en_offset_x = (max_width - params.en_roi_width) / 2;
    if (params.en_offset_y == -1) params.en_offset_y = (max_height - params.en_roi_height) / 2;
    RCLCPP_INFO(this->get_logger(), "en_roi_height: %d, en_roi_width: %d", params.en_roi_height,
                params.en_roi_width);
    RCLCPP_INFO(this->get_logger(), "en_offset_x: %d, en_offset_y: %d", params.en_offset_x,
                params.en_offset_y);

    set_grab_params(params.offset_x, params.offset_y, params.roi_width, params.roi_height);
    UPDBW(MV_CC_SetEnumValue(camera_handle, "TriggerMode", MV_TRIGGER_MODE_OFF))
    UPDBW(MV_CC_SetEnumValue(camera_handle, "ExposureMode", MV_EXPOSURE_AUTO_MODE_OFF))
    UPDBW(MV_CC_SetEnumValue(camera_handle, "GainAuto", MV_GAIN_MODE_OFF))
    UPDBW(MV_CC_SetBoolValue(camera_handle, "BlackLevelEnable", true))
    UPDBW(MV_CC_SetEnumValue(camera_handle, "BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_ONCE))
    UPDBW(MV_CC_SetEnumValue(camera_handle, "AcquisitionMode", MV_ACQ_MODE_CONTINUOUS))
    UPDBW(MV_CC_SetFloatValue(camera_handle, "AcquisitionFrameRate", params.frame_rate))
    UPDBW(MV_CC_SetBoolValue(camera_handle, "AcquisitionFrameRateEnable", true))
    UPDBW(MV_CC_SetFloatValue(camera_handle, "ExposureTime", params.exposure_time))
    UPDBW(MV_CC_SetFloatValue(camera_handle, "Gain", params.gain))
    // UPDBW(MV_CC_SetBoolValue(camera_handle, "GammaEnable", true))
    // UPDBW(MV_CC_SetFloatValue(camera_handle, "Gamma", params.gamma))
    UPDBW(MV_CC_SetBoolValue(camera_handle, "DigitalShiftEnable", true))
    UPDBW(MV_CC_SetFloatValue(camera_handle, "DigitalShift", params.digital_shift))
}

void HikCameraNode::robot_mode_update(rm_interfaces::msg::Rmrobot::ConstSharedPtr msg) {
    vision_mode now_vision_mode = string2mode(msg->vision_mode);
    params.robot_id = (Robot_id_dji)msg->robot_id;
    params.bullet_velocity = msg->bullet_velocity;
    params.right_press = msg->right_press;
    params.lobshot = msg->lobshot;
    if (params.mode != now_vision_mode) {
        RCLCPP_WARN(get_logger(), "Vision mode change to %s!", msg->vision_mode.c_str());
    }
    if (params.mode == AUTO_AIM && (now_vision_mode == B_WM || now_vision_mode == S_WM)) {
        RCLCPP_WARN(get_logger(), "Vision mode change to energy!");
        params.mode = now_vision_mode;
        stop_grab();
        set_grab_params(params.en_offset_x, params.en_offset_y, params.en_roi_width,
                        params.en_roi_height);
        grab_vision_mode = now_vision_mode;
        start_grab();
    } else if ((params.mode == B_WM || params.mode == S_WM) && now_vision_mode == AUTO_AIM) {
        RCLCPP_WARN(get_logger(), "Vision mode change to normal(armor)!");
        stop_grab();
        set_grab_params(params.offset_x, params.offset_y, params.roi_width, params.roi_height);
        grab_vision_mode = now_vision_mode;
        start_grab();
    }
    params.mode = now_vision_mode;
}

void HikCameraNode::grab() {
    MV_FRAME_OUT out_frame;
    RCLCPP_INFO(this->get_logger(), "Publishing image!");

    // Init convert param
    convert_param.nWidth = img_info.nWidthValue;
    convert_param.nHeight = img_info.nHeightValue;
    convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed;

    // 创建消息
    sensor_msgs::msg::Image image_msg;
    image_msg.encoding = "bgr8";
    image_msg.height = img_info.nHeightValue;
    image_msg.width = img_info.nWidthValue;
    image_msg.step = img_info.nWidthValue * 3;
    image_msg.data.resize(img_info.nWidthValue * img_info.nHeightValue * 3);

    // 创建消息Resize_640()
    sensor_msgs::msg::Image image_msg_resize;
    image_msg_resize.encoding = "bgr8";
    image_msg_resize.height = 640;
    image_msg_resize.width = 640;
    image_msg_resize.step = 640 * 3;
    image_msg_resize.data.resize(640 * 640 * 3);

    // frame_info_cam
    FrameInfo cam_frame_info;
    // zero-copy mat
    cv::Mat img_640(image_msg_resize.height, image_msg_resize.width,
                    encoding2mat_type(image_msg_resize.encoding), image_msg_resize.data.data());

    while (rclcpp::ok() && grab_on) {
        PerfGuard hik_cam_perf("HikCameraGrab");
        nRet = MV_CC_GetImageBuffer(camera_handle, &out_frame, 1000);
        if (MV_OK == nRet) {
            // sensor_msgs::msg::Image::UniquePtr image_msg(new sensor_msgs::msg::Image());

            convert_param.pDstBuffer = image_msg.data.data();
            convert_param.nDstBufferSize = image_msg.data.size();
            convert_param.pSrcData = out_frame.pBufAddr;
            convert_param.nSrcDataLen = out_frame.stFrameInfo.nFrameLen;
            convert_param.enSrcPixelType = out_frame.stFrameInfo.enPixelType;

            MV_CC_ConvertPixelType(camera_handle, &convert_param);
            auto time_now = this->now();
            // load cam_frame_info
            cam_frame_info.mode = grab_vision_mode;
            cam_frame_info.d = camera_info_msg.d;
            cam_frame_info.k =
                std::vector<double>(camera_info_msg.k.begin(), camera_info_msg.k.end());
            cam_frame_info.robot_id = params.robot_id;
            cam_frame_info.bullet_velocity = params.bullet_velocity;
            cam_frame_info.right_press = params.right_press;
            cam_frame_info.lobshot = params.lobshot;

            // 根据corp调整cx/cy
            if (grab_vision_mode == B_WM || grab_vision_mode == S_WM) {
                cam_frame_info.k[2] -= params.en_offset_x;
                cam_frame_info.k[5] -= params.en_offset_y;
            } else {
                cam_frame_info.k[2] -= params.offset_x;
                cam_frame_info.k[5] -= params.offset_y;
            }
            cv::Mat img_origin(image_msg.height, image_msg.width,
                               encoding2mat_type(image_msg.encoding), image_msg.data.data());
            if ((grab_vision_mode == B_WM || grab_vision_mode == S_WM) && params.en_resize_640) {
                double rx = 640.0 / (double)image_msg.width;
                double ry = 640.0 / (double)image_msg.height;
                cam_frame_info.k[0] *= rx;
                cam_frame_info.k[2] *= rx;
                cam_frame_info.k[4] *= ry;
                cam_frame_info.k[5] *= ry;
                // zero-copy mat

                cv::resize(img_origin, img_640, cv::Size(640, 640), cv::INTER_NEAREST);
                image_msg_resize.header.stamp = time_now;
                image_msg_resize.header.frame_id = cam_frame_info.serialize();
                cv::LUT(img_640, Gamma_lookUpTable, img_640);
                image_pub.publish(image_msg_resize);
            } else {
                image_msg.header.stamp = time_now;
                image_msg.header.frame_id = cam_frame_info.serialize();
                {
                    PerfGuard LUT("LUT");
                    cv::LUT(img_origin, Gamma_lookUpTable, img_origin);
                }
                image_pub.publish(image_msg);
                // RCLCPP_INFO(get_logger(),"FID: %s",cam_frame_info.serialize().c_str());
            }

            MV_CC_FreeImageBuffer(camera_handle, &out_frame);
            fail_cnt = 0;
        } else {
            RCLCPP_WARN(this->get_logger(), "Get buffer failed! nRet: [%x]", nRet);
            // MV_CC_StopGrabbing(camera_handle);
            // MV_CC_StartGrabbing(camera_handle);
            fail_cnt++;
        }

        if (fail_cnt > 5) {
            RCLCPP_FATAL(this->get_logger(), "Camera failed!");
            grab_on = false;
            camera_failed = true;
        }
    }
}

void HikCameraNode::reset() {
    close_device();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    init_camera();
}

void HikCameraNode::open_device() {
    UPDBE(MV_CC_OpenDevice(camera_handle));
    UPDBE(MV_CC_CloseDevice(camera_handle));
    // 来回开关, 确保相机状态正常(增强鲁棒性措施)
    UPDBE(MV_CC_OpenDevice(camera_handle));
}

void HikCameraNode::close_device() {
    stop_grab();
    if (camera_handle) {
        MV_CC_CloseDevice(camera_handle);
        MV_CC_DestroyHandle(camera_handle);
    }
    RCLCPP_INFO(this->get_logger(), "HikCamera closed!");
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(hik_camera::HikCameraNode)
