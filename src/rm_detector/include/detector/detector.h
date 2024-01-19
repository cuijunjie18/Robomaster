#ifndef _DETECTOR_H_
#define _DETECTOR_H_

#include <detector/net_decoder.h>
#include <rm_utils/data.h>

#include <rm_utils/perf.hpp>
// ROS
#include <opencv2/opencv.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>

class Detector {
   protected:
    rclcpp::Logger logger;
    std::string type;
    toml::value config;
    Detector(const std::string &config_file, const std::string &share_dir,
             const rclcpp::Logger &_logger)
        : logger(_logger) {
        config = toml::parse(share_dir + "/" + config_file);
    };

    cv::Mat static_resize(cv::Mat img, int INPUT_H, int INPUT_W);

   public:
    virtual std::vector<Armor> detect(cv::Mat);
    virtual void draw(cv::Mat, const std::vector<Armor> &);
};

class ParallelImg2Blob : public cv::ParallelLoopBody {
   private:
    const cv::Mat &img;
    float *blob_data;
    int img_h;
    int img_w;

   public:
    ParallelImg2Blob(const cv::Mat &img, float *blob_data)
        : img(img), blob_data(blob_data), img_h(img.rows), img_w(img.cols) {}

    virtual void operator()(const cv::Range &range) const override {
        for (int r = range.start; r < range.end; r++) {
            const uchar *uc_pixel = img.ptr<uchar>(r);
            for (int col = 0; col < img_w; col++) {
                int i = r * img_w + col;
                blob_data[i] = (float)uc_pixel[2] / 255.0;
                blob_data[i + img_h * img_w] = (float)uc_pixel[1] / 255.0;
                blob_data[i + 2 * img_h * img_w] = (float)uc_pixel[0] / 255.0;
                uc_pixel += 3;
            }
        }
    }
};

class NetDetector : public Detector {
   protected:
    // for drawing
    std::vector<std::string> class_names, color_names, tsize_names;
    int INPUT_W, INPUT_H, NUM_CLASSES, NUM_COLORS;
    float BBOX_CONF_THRESH, NMS_THRESH, MERGE_THRESH;
    int point_num;

    int layer_num;
    std::shared_ptr<NetDecoderBase> decoder;
    std::string model_prefix;

    std::vector<Armor> do_nms(std::vector<Armor> &);
    std::vector<Armor> do_merge_nms(std::vector<Armor> &);
    void img2blob(const cv::Mat &img, float *);
    NetDetector(const std::string &config_file, const std::string &share_dir,
                const rclcpp::Logger &_logger);

   public:
    void draw(cv::Mat, const std::vector<Armor> &) override;
};

#endif