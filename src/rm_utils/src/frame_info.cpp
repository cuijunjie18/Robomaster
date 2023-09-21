#include <rm_utils/frame_info.h>

#include <sstream>
#include <toml.hpp>
#include <yaml-cpp/yaml.h>
std::string FrameInfo::serialize() {
    YAML::Node v;
    v["vision_mode"] = (int)mode;
    v["robot_id"] = (int)robot_id;
    v["right_press"] = right_press;
    v["lobshot"] = lobshot;
    v["v"] = bullet_velocity;
    v["k"] = k;
    v["d"] = d;
    YAML::Emitter out;
    out << v;
    return out.c_str();
}

void FrameInfo::deserialize(const std::string& data) {
    YAML::Node v = YAML::Load(data);
    mode = (vision_mode)v["vision_mode"].as<int>();
    robot_id = (Robot_id_dji)v["robot_id"].as<int>();
    right_press = v["right_press"].as<bool>();
    lobshot = v["lobshot"].as<bool>();
    bullet_velocity = v["v"].as<double>();
    k = v["k"].as<std::vector<double>>();
    d = v["d"].as<std::vector<double>>();
}