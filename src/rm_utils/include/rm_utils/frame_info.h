#ifndef _FRAME_INFO_H
#define _FRAME_INFO_H

#include <rm_utils/data.h>
#include <vector>
#include <string>

// 随每一帧图像发送，包含系统关键信息
class FrameInfo {
   public:
    vision_mode mode;
    bool right_press;
    bool lobshot;
    Robot_id_dji robot_id;
    double bullet_velocity;
    std::vector<double> k;
    std::vector<double> d;
    std::string serialize();
    void deserialize(const std::string&);
};

#endif
