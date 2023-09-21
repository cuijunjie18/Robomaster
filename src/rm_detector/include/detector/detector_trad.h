#ifndef CRH_DETECT_TRAD_HPP_
#define CRH_DETECT_TRAD_HPP_

#include <rm_interfaces/msg/debug_armors.hpp>
#include <rm_interfaces/msg/debug_lights.hpp>

#include <detector/detector.h>
#include <opencv2/opencv.hpp>

class TraditonalDetectorArgs {
    void init(const toml::value&, const std::string&);

   public:
    std::string path2model_mlp;
    float min_lightness;
    float mlp_threshold;

    struct {
        float max_angle;
        float max_ratio;
        float min_ratio;
    } light_args;

    struct {
        float max_angle;
        float max_center_ratio;
        float min_center_ratio;
        float min_light_ratio;
        float max_large_center_distance;
        float min_large_center_distance;
        float max_small_center_distance;
        float min_small_center_distance;
    } armor_args;
    friend class DetectorTrad;
};

struct Light : public cv::RotatedRect {
    Light() = default;
    explicit Light(cv::RotatedRect box) : cv::RotatedRect(box) {
        cv::Point2f p[4];
        box.points(p);
        std::sort(p, p + 4, [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
        top = (p[0] + p[1]) / 2;
        bottom = (p[2] + p[3]) / 2;

        length = cv::norm(top - bottom);
        width = cv::norm(p[0] - p[1]);

        tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
        tilt_angle = tilt_angle / CV_PI * 180;
    }

    int color;
    cv::Point2f top, bottom;
    double length;
    double width;
    float tilt_angle;
};

struct ArmorTrad {
    enum ArmorType { SMALL = 0, LARGE = 1, INVALID = 2};
    ArmorTrad() = default;
    ArmorTrad(const Light& l1, const Light& l2) {
        if (l1.center.x < l2.center.x) {
            left_light = l1, right_light = l2;
        } else {
            left_light = l2, right_light = l1;
        }
        center = (left_light.center + right_light.center) / 2;
    }
    Light left_light, right_light;
    cv::Point2f center;

    cv::Mat number_img;

    int class_id;
    std::string class_name;
    float similarity;
    float confidence;
    std::string classfication_result;
    ArmorType armor_type;
};

class DetectorTrad : public Detector {
    TraditonalDetectorArgs params;
    cv::Mat src_bgr;
    cv::Mat gray_img;

    cv::dnn::Net mlp_net;

    std::vector<Light> lights;
    std::vector<ArmorTrad> armors;
    std::vector<std::string> class_names;

    static constexpr int map_2_standard[9] = {1, 2, 3, 4, 5, 6, 0, 7, -1};

    bool isLight(const Light& light);
    bool isArmor(ArmorTrad& armor);
    bool containLight(const Light& light_1, const Light& light_2);
    void preprocessImage();
    void findLights();
    void matchLights();
    void extractNumbers();
    void classifyNums();
    

   public:
    // debug msg
    cv::Mat binary_img;
    rm_interfaces::msg::DebugLights debug_lights;
    rm_interfaces::msg::DebugArmors debug_armors;
    cv::Mat getAllNumbersImage();
    enum ColorType { RED = 1, BLUE = 0 };
    DetectorTrad(const std::string& file_name, const std::string& share_dir,
                 const rclcpp::Logger& _logger);
    std::vector<Armor> detect(cv::Mat _src) override;
    void draw(cv::Mat, const std::vector<Armor>&) override;
};
#endif