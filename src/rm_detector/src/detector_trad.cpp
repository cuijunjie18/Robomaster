#include <detector/detector_trad.h>

void TraditonalDetectorArgs::init(const toml::value& config, const std::string& share_dir) {
    auto& armor = config.at("armor");
    auto& light = config.at("light");
    path2model_mlp = share_dir + "/" + std::string(config.at("mlp_path").as_string());

    mlp_threshold = config.at("mlp_threshold").as_floating();
    min_lightness = config.at("min_lightness").as_floating();

    light_args.max_angle = light.at("max_angle").as_floating();
    light_args.max_ratio = light.at("max_ratio").as_floating();
    light_args.min_ratio = light.at("min_ratio").as_floating();

    armor_args.max_angle = armor.at("max_angle").as_floating();
    armor_args.max_center_ratio = armor.at("max_center_ratio").as_floating();
    armor_args.min_center_ratio = armor.at("min_center_ratio").as_floating();
    armor_args.min_light_ratio = armor.at("min_light_ratio").as_floating();
    armor_args.max_large_center_distance = armor.at("max_large_center_distance").as_floating();
    armor_args.min_large_center_distance = armor.at("min_large_center_distance").as_floating();
    armor_args.max_small_center_distance = armor.at("max_small_center_distance").as_floating();
    armor_args.min_small_center_distance = armor.at("min_small_center_distance").as_floating();
}

DetectorTrad::DetectorTrad(const std::string& config_file, const std::string& share_dir,
                           const rclcpp::Logger& _logger)
    : Detector(config_file, share_dir, _logger) {
    auto config = toml::parse(share_dir + "/" + config_file);

    type = config.at("type").as_string();

    params.init(config, share_dir);

    mlp_net = cv::dnn::readNetFromONNX(params.path2model_mlp);
    class_names = toml::get<std::vector<std::string>>(config.at("class_names"));
    RCLCPP_INFO(logger, "[TRAD] Tradition Detector Init done.");
}

void DetectorTrad::preprocessImage() {
    cv::cvtColor(src_bgr, gray_img, cv::COLOR_BGR2GRAY);
    cv::threshold(gray_img, binary_img, params.min_lightness, 255, cv::THRESH_BINARY);
    // cv::imshow("bin", binary_img);
}

bool DetectorTrad::isLight(const Light& light) {
    // The ratio of light (short side / long side)
    float ratio = light.width / light.length;
    bool ratio_ok = params.light_args.min_ratio < ratio && ratio < params.light_args.max_ratio;
    bool angle_ok = light.tilt_angle < params.light_args.max_angle;
    bool is_light = ratio_ok && angle_ok;

    // Fill in debug information
    rm_interfaces::msg::DebugLight light_data;
    light_data.center_x = light.center.x;
    light_data.ratio = ratio;
    light_data.angle = light.tilt_angle;
    light_data.is_light = is_light;
    this->debug_lights.data.emplace_back(light_data);

    return is_light;
}

void DetectorTrad::findLights() {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    lights.clear();
    this->debug_lights.data.clear();
    for (const auto& contour : contours) {
        if (contour.size() < 5) continue;

        auto r_rect = cv::minAreaRect(contour);
        auto light = Light(r_rect);

        if (isLight(light)) {
            auto rect = light.boundingRect();
            if (  // Avoid assertion failed
                0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= src_bgr.cols &&
                0 <= rect.y && 0 <= rect.height && rect.y + rect.height <= src_bgr.rows) {
                int sum_r = 0, sum_b = 0;
                auto roi = src_bgr(rect);
                // Iterate through the ROI
                for (int i = 0; i < roi.rows; i++) {
                    for (int j = 0; j < roi.cols; j++) {
                        if (cv::pointPolygonTest(contour, cv::Point2f(j + rect.x, i + rect.y),
                                                 false) >= 0) {
                            // if point is inside contour
                            sum_r += roi.at<cv::Vec3b>(i, j)[2];
                            sum_b += roi.at<cv::Vec3b>(i, j)[0];
                        }
                    }
                }
                // Sum of red pixels > sum of blue pixels ?
                light.color = sum_r > sum_b ? RED : BLUE;
                lights.emplace_back(light);
            }
        }
    }
}

// Check if there is another light in the boundingRect formed by the 2 lights
bool DetectorTrad::containLight(const Light& light_1, const Light& light_2) {
    auto points =
        std::vector<cv::Point2f>{light_1.top, light_1.bottom, light_2.top, light_2.bottom};
    auto bounding_rect = cv::boundingRect(points);

    for (const auto& test_light : lights) {
        if (test_light.center == light_1.center || test_light.center == light_2.center) continue;

        if (bounding_rect.contains(test_light.top) || bounding_rect.contains(test_light.bottom) ||
            bounding_rect.contains(test_light.center)) {
            return true;
        }
    }

    return false;
}

bool DetectorTrad::isArmor(ArmorTrad& armor) {
    Light light_1 = armor.left_light;
    Light light_2 = armor.right_light;
    // Ratio of the length of 2 lights (short side / long side)
    float light_length_ratio = light_1.length < light_2.length ? light_1.length / light_2.length
                                                               : light_2.length / light_1.length;
    bool light_ratio_ok = light_length_ratio > params.armor_args.min_light_ratio;

    // Distance between the center of 2 lights (unit : light length)
    float avg_light_length = (light_1.length + light_2.length) / 2;
    float center_distance = cv::norm(light_1.center - light_2.center) / avg_light_length;
    bool center_distance_ok = (params.armor_args.min_small_center_distance < center_distance &&
                               center_distance < params.armor_args.max_small_center_distance) ||
                              (params.armor_args.min_large_center_distance < center_distance &&
                               center_distance < params.armor_args.max_large_center_distance);

    // Angle of light center connection
    cv::Point2f diff = light_1.center - light_2.center;
    float angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
    bool angle_ok = angle < params.armor_args.max_angle;

    bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;
    armor.armor_type = center_distance > params.armor_args.min_large_center_distance
                           ? ArmorTrad::LARGE
                           : ArmorTrad::SMALL;
    if (!is_armor) armor.armor_type = ArmorTrad::INVALID;

    // Fill in debug information
    rm_interfaces::msg::DebugArmor armor_data;
    armor_data.type = armor.armor_type == ArmorTrad::LARGE   ? "LARGE"
                      : armor.armor_type == ArmorTrad::SMALL ? "SMALL"
                                                             : "INVALID";
    armor_data.center_x = (light_1.center.x + light_2.center.x) / 2;
    armor_data.light_ratio = light_length_ratio;
    armor_data.center_distance = center_distance;
    armor_data.angle = angle;
    this->debug_armors.data.emplace_back(armor_data);

    return is_armor;
}

void DetectorTrad::matchLights() {
    armors.clear();
    this->debug_armors.data.clear();
    // Loop all the pairing of lights
    for (auto light_1 = lights.begin(); light_1 != lights.end(); light_1++) {
        for (auto light_2 = light_1 + 1; light_2 != lights.end(); light_2++) {
            if (light_1->color != light_2->color) continue;  // 同色匹配

            if (containLight(*light_1, *light_2)) {
                continue;
            }
            auto armor = ArmorTrad(*light_1, *light_2);
            if (isArmor(armor)) {
                armors.emplace_back(armor);
            }
        }
    }
}

void DetectorTrad::extractNumbers() {
    // Light length in image
    const int light_length = 12;
    // Image size after warp
    const int warp_height = 28;
    const int small_armor_width = 32;
    const int large_armor_width = 54;
    // Number ROI size
    const cv::Size roi_size(20, 28);

    for (auto& armor : armors) {
        // Warp perspective transform
        cv::Point2f lights_vertices[4] = {armor.left_light.bottom, armor.left_light.top,
                                          armor.right_light.top, armor.right_light.bottom};

        const int top_light_y = (warp_height - light_length) / 2 - 1;
        const int bottom_light_y = top_light_y + light_length;
        const int warp_width =
            armor.armor_type == ArmorTrad::SMALL ? small_armor_width : large_armor_width;
        cv::Point2f target_vertices[4] = {
            cv::Point(0, bottom_light_y),
            cv::Point(0, top_light_y),
            cv::Point(warp_width - 1, top_light_y),
            cv::Point(warp_width - 1, bottom_light_y),
        };
        cv::Mat number_image;
        auto rotation_matrix = cv::getPerspectiveTransform(lights_vertices, target_vertices);
        cv::warpPerspective(src_bgr, number_image, rotation_matrix,
                            cv::Size(warp_width, warp_height));

        // Get ROI
        number_image =
            number_image(cv::Rect(cv::Point((warp_width - roi_size.width) / 2.0, 0), roi_size));

        // Binarize
        cv::cvtColor(number_image, number_image, cv::COLOR_RGB2GRAY);
        cv::threshold(number_image, number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        armor.number_img = number_image;
    }
}

void DetectorTrad::classifyNums() {
    for (auto& armor : armors) {
        cv::Mat image = armor.number_img.clone();

        // Normalize
        image = image / 255.0;

        // Create blob from image
        cv::Mat blob;
        cv::dnn::blobFromImage(image, blob);

        // Set the input blob for the neural network
        mlp_net.setInput(blob);
        // Forward pass the image blob through the model
        cv::Mat outputs = mlp_net.forward();

        // Do softmax
        float max_prob = *std::max_element(outputs.begin<float>(), outputs.end<float>());
        cv::Mat softmax_prob;
        cv::exp(outputs - max_prob, softmax_prob);
        float sum = static_cast<float>(cv::sum(softmax_prob)[0]);
        softmax_prob /= sum;

        double confidence;
        cv::Point class_id_point;
        minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);

        armor.class_id = class_id_point.x;
        armor.confidence = confidence;
        armor.class_name = class_names[armor.class_id];
        // RCLCPP_INFO(logger, "[TRAD] armor: %d %f %s", armor.class_id, armor.confidence,
        //             armor.class_name.c_str());
        std::stringstream result_ss;
        result_ss << armor.class_name << ": " << std::fixed << std::setprecision(1)
                  << armor.confidence * 100.0 << "%";
        armor.classfication_result = result_ss.str();
    }

    armors.erase(std::remove_if(armors.begin(), armors.end(),
                                [this](const ArmorTrad& armor) {
                                    if (armor.confidence < params.mlp_threshold ||
                                        armor.class_name == "Negative") {
                                        return true;
                                    }

                                    // if (armor.class_name == "Outpost"){
                                    //     return true;
                                    // }

                                    bool mismatch_armor_type = false;
                                    if (armor.armor_type == ArmorTrad::LARGE) {
                                        mismatch_armor_type = armor.class_name == "Outpost" ||
                                                              armor.class_name == "2" ||
                                                              armor.class_name == "Guard";
                                    } else if (armor.armor_type == ArmorTrad::SMALL) {
                                        mismatch_armor_type =
                                            armor.class_name == "1" || armor.class_name == "Base";
                                    }
                                    return mismatch_armor_type;
                                }),
                 armors.end());
}

cv::Mat DetectorTrad::getAllNumbersImage() {
    if (armors.empty()) {
        return cv::Mat(cv::Size(20, 28), CV_8UC1);
    } else {
        std::vector<cv::Mat> number_imgs;
        number_imgs.reserve(armors.size());
        for (auto& armor : armors) {
            number_imgs.emplace_back(armor.number_img);
        }
        cv::Mat all_num_img;
        cv::vconcat(number_imgs, all_num_img);
        return all_num_img;
    }
}

std::vector<Armor> DetectorTrad::detect(cv::Mat _src) {
    // logger.info("src shape {} x {}", _src.rows, _src.cols);
    if (_src.empty()) return std::vector<Armor>();
    src_bgr = _src;  // 浅拷贝
    preprocessImage();
    findLights();
    matchLights();
    extractNumbers();
    classifyNums();
    std::vector<Armor> dets;
    dets.clear();

    for (auto& armor_trad : armors) {
        Armor now;
        now.color = armor_trad.left_light.color;
        now.conf = armor_trad.confidence;
        now.pts[0] = armor_trad.left_light.top;
        now.pts[1] = armor_trad.left_light.bottom;
        now.pts[2] = armor_trad.right_light.bottom;
        now.pts[3] = armor_trad.right_light.top;
        now.rect = cv::boundingRect(
            std::vector<cv::Point2f>({now.pts[0], now.pts[1], now.pts[2], now.pts[3]}));
        now.type = map_2_standard[armor_trad.class_id];
        now.size = armor_trad.armor_type;
        dets.push_back(now);
    }
    // cv::imshow("numbers", getAllNumbersImage());
    // cv::waitKey(1);
    return dets;
}

void DetectorTrad::draw(cv::Mat img, const std::vector<Armor>&) {
    // Draw Lights
    for (const auto& light : lights) {
        cv::circle(img, light.top, 3, cv::Scalar(255, 255, 255), 1);
        cv::circle(img, light.bottom, 3, cv::Scalar(255, 255, 255), 1);
        auto line_color = light.color == RED ? cv::Scalar(255, 255, 0) : cv::Scalar(255, 0, 255);
        cv::line(img, light.top, light.bottom, line_color, 1);
    }

    // Draw armors
    for (const auto& armor : armors) {
        cv::line(img, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
        cv::line(img, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
    }

    // Show numbers and confidence
    for (const auto& armor : armors) {
        cv::putText(img, armor.classfication_result, armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX,
                    0.8, cv::Scalar(0, 255, 255), 2);
    }
}