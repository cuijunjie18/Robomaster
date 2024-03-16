#ifndef _RMUTILS_DATA_H
#define _RMUTILS_DATA_H
#define UNKNOWN_ID (15)
#include <rm_interfaces/msg/armor.hpp>
#include <rm_interfaces/msg/control.hpp>

typedef rm_interfaces::msg::Armor ArmorMsg;
typedef rm_interfaces::msg::Control ControlMsg;

#if __has_include(<opencv2/opencv.hpp>)

#include <opencv2/opencv.hpp>
struct Armor {
    int type;   // armor_type
    int color;  // BRNP
    int size;   // 0小1大
    float conf;
    cv::Rect rect;
    cv::Point2f pts[5];
};

ArmorMsg Armor2Msg(const Armor& armor);
Armor Msg2Armor(const ArmorMsg& armor_msg);

#endif

// 视觉模式
enum vision_mode {
    NO_AIM,           // 无瞄准
    AUTO_AIM,         // 普通自瞄
    ANTI_ROT,         // 反陀螺
    B_WM,             // 大风车
    S_WM,             // 小风车
    HALT,             // 停机
    Unknown,          // 未知
};

// 机器人ID（官方定义/电控定义）
typedef enum Robot_id_e {
    RED_HERO = 1,
    RED_ENGINEER = 2,
    RED_STANDARD_1 = 3,
    RED_STANDARD_2 = 4,
    RED_STANDARD_3 = 5,
    RED_AERIAL = 6,
    RED_SENTRY = 7,
    BLUE_HERO = 101,
    BLUE_ENGINEER = 102,
    BLUE_STANDARD_1 = 103,
    BLUE_STANDARD_2 = 104,
    BLUE_STANDARD_3 = 105,
    BLUE_AERIAL = 106,
    BLUE_SENTRY = 107,
} Robot_id_dji;

// 简化机器人ID（没有颜色）
enum Robot_id_simple {
    ROBOT_SENTRY = 0,
    ROBOT_HERO = 1,
    ROBOT_ENGINEER = 2,
    ROBOT_STANDARD_1 = 3,
    ROBOT_STANDARD_2 = 4,
    ROBOT_STANDARD_3 = 5,
    ROBOT_AERIAL = 6,
};

// 装甲板ID（RMCV定义）
enum armor_type {
    SENTRY = 0,
    HERO = 1,
    ENGINEER = 2,
    STANDARD_1 = 3,
    STANDARD_2 = 4,
    STANDARD_3 = 5,
    OUTPOST = 6,
    BASE = 7,
    TOP = 8,
};

#pragma pack(1)
typedef struct recv_msg_ {
    float roll;
    float pitch;
    float yaw;
    /* mode定义
        0: 关闭自瞄（步兵、无人机松开右键）（英雄按F切换自瞄开关）
        1: 普通自瞄 (英雄带有击打前哨站旋转装甲板能力)
        2: 小风车模式（步兵专用）
        3: 大风车模式（步兵专用）
        4: 吊射模式（击打前哨站和基地的顶装甲板）（英雄专用）
        */

    uint8_t mode;
    // id定义：与官方定义相同
    uint8_t robot_id;
    // v定义:实际弹速
    float bullet_speed;
} recv_msg;

typedef struct send_msg_ {
    float pitch;
    float yaw;
    uint8_t one_shot_num;
    uint8_t rate;
    uint8_t vision_follow_id;
} send_msg;

typedef struct serial_header_ {
    uint8_t stater;
    uint16_t id;
    uint8_t len;
} serial_header;

typedef struct auto_shoot_from_pc_t {
    // pitch yaw 目标位置，删去roll
    float pitch;
    float yaw;
    // 发射n颗弹，记为一次发射行为
    uint8_t one_shot_num;
    // 一秒中发射行为的频率
    uint8_t rate;
    // 瞄准的机器人id
    uint8_t vision_follow_id;
    auto_shoot_from_pc_t(float _pitch, float _yaw, int8_t _one_shot_num, int8_t _rate, int8_t _vision_follow_id)
        : pitch(_pitch), yaw(_yaw), one_shot_num(_one_shot_num), rate(_rate), vision_follow_id(_vision_follow_id) {}
} auto_shoot_from_pc_t;

#pragma pack()

uint8_t get_rmcv_id(uint8_t id);
vision_mode cast_run_mode(uint8_t mode);
vision_mode string2mode(const std::string& mode_str);
std::string mode2string(uint8_t mode);
ControlMsg make_cmd(auto_shoot_from_pc_t shoot_behavior);
#endif
