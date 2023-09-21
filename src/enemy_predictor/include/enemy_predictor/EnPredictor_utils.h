#ifndef HITSZ_RMVISION_PREDICT_UTILS_HPP_
#define HITSZ_RMVISION_PREDICT_UTILS_HPP_

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <rm_utils/ballistic.hpp>

#define AMeps 1e-2

class Filter {
   private:
    double sum;
    std::queue<double> data;
    size_t max_length;

   public:
    Filter(size_t _max_length = 100) : sum(0.), data(), max_length(_max_length) {}
    Filter(const Filter &) = default;

    double get() {
        if (data.empty()) {
            return 0.;
        }
        return sum / data.size();
    }
    void update(const double &item) {
        if (data.size() == max_length) {
            sum -= data.front();
            data.pop();
        }
        sum += item;
        data.push(item);
    }
};

// 工具函数

inline double get_disAngle(double ag1, double ag2) {
    double diff = fmod(ag1 - ag2, M_PI * 2);
    if (diff > M_PI) {
        return diff - 2 * M_PI;
    } else if (diff < -M_PI) {
        return diff + 2 * M_PI;
    } else {
        return diff;
    }
}

// 计算三维点距离
inline double get_dis3d(Eigen::Matrix<double, 3, 1> A,
                        Eigen::Matrix<double, 3, 1> B = Eigen::Matrix<double, 3, 1>::Zero()) {
    return (A[0] - B[0]) * (A[0] - B[0]) + (A[1] - B[1]) * (A[1] - B[1]) +
           (A[2] - B[2]) * (A[2] - B[2]);
}
// 计算二维点距离
inline double get_dis2d(cv::Point2d a, cv::Point2d b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}
// 将云台旋转至世界坐标系
inline Eigen::Matrix<double, 3, 3> get_rot_gripper2base(double roll, double pitch, double yaw) {
    Eigen::Matrix<double, 3, 3> Rx, Ry, Rz;
    Rx << 1., 0., 0., 0., cos(roll), -sin(roll), 0., sin(roll), cos(roll);
    Ry << cos(pitch), 0., sin(pitch), 0., 1., 0., -sin(pitch), 0., cos(pitch);
    Rz << cos(yaw), -sin(yaw), 0., sin(yaw), cos(yaw), 0., 0., 0., 1.;
    return Rz * Ry * Rx;
}
// 叉乘 计算三角形面积的两倍
inline double cross(cv::Point2d a, cv::Point2d b) { return a.x * b.y - a.y * b.x; }
inline double get_S_triangle(cv::Point2d a, cv::Point2d b, cv::Point2d c) {
    return fabs(cross(b - a, c - a)) / 2;
}
inline double get_area_armor(cv::Point2f pts[5]) {
    return get_S_triangle(pts[0], pts[1], pts[2]) + get_S_triangle(pts[3], pts[0], pts[2]);
}
inline double IOU(const cv::Rect_<float> &box1, const cv::Rect_<float> &box2) {
    if (box1.x > box2.x + box2.width) {
        return 0.0;
    }
    if (box1.y > box2.y + box2.height) {
        return 0.0;
    }
    if (box1.x + box1.width < box2.x) {
        return 0.0;
    }
    if (box1.y + box1.height < box2.y) {
        return 0.0;
    }
    float colInt = std::min(box1.x + box1.width, box2.x + box2.width) - std::max(box1.x, box2.x);
    float rowInt = std::min(box1.y + box1.height, box2.y + box2.height) - std::max(box1.y, box2.y);
    float intersection = colInt * rowInt;
    float area1 = box1.width * box1.height;
    float area2 = box2.width * box2.height;
    return intersection / (area1 + area2 - intersection);
}
// 检测世界坐标系下两点坐标相对位置
inline bool check_relative_pos(Eigen::Matrix<double, 3, 1> a, Eigen::Matrix<double, 3, 1> b) {
    return a[0] * b[1] > a[1] * b[0];
}
/// 计算三维向量之间的夹角
inline double calc_diff_angle_xyz(const Eigen::Matrix<double, 3, 1> &a,
                                  const Eigen::Matrix<double, 3, 1> &b) {
    return fabs(acos(a.dot(b) / a.norm() / b.norm()));
}
/// 计算球面距离（减少distance误差影响）
inline double calc_surface_dis_xyz(const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
    return calc_diff_angle_xyz(a, b) * (a.norm() + b.norm()) / 2;
}

inline bool check_left(const Eigen::Matrix<double, 3, 1> &pyd_left,
                       const Eigen::Matrix<double, 3, 1> &pyd_right) {
    double yaw_l = pyd_left[1];
    double yaw_r = pyd_right[1];
    double yaw_dis = fmod(yaw_l - yaw_r, M_PI * 2);
    if (fabs(yaw_dis) > M_PI)
        return yaw_dis < 0;
    else
        return yaw_dis > 0;
}

inline Eigen::Matrix<double, 3, 1> xyz2pyd(const Eigen::Matrix<double, 3, 1> &xyz) {
    Eigen::Matrix<double, 3, 1> pyd;
    pyd[0] = -atan2(xyz[2], sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]));
    pyd[1] = atan2(xyz[1], xyz[0]);
    pyd[2] = sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]);
    return pyd;
}

inline Eigen::Matrix<double, 3, 1> pyd2xyz(const Eigen::Matrix<double, 3, 1> &pyd) {
    Eigen::Matrix<double, 3, 1> xyz;
    xyz[2] = -pyd[2] * sin(pyd[0]);
    double tmplen = pyd[2] * cos(pyd[0]);
    xyz[1] = tmplen * sin(pyd[1]);
    xyz[0] = tmplen * cos(pyd[1]);
    return xyz;
}

inline Eigen::Matrix<double, 3, 1> xyz2dyz(const Eigen::Matrix<double, 3, 1> &xyz) {
    Eigen::Matrix<double, 3, 1> dyz;
    dyz[0] = sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]);
    dyz[1] = atan2(xyz[1], xyz[0]);
    dyz[2] = xyz[2];
    return dyz;
}

inline Eigen::Matrix<double, 3, 1> dyz2xyz(const Eigen::Matrix<double, 3, 1> &dyz) {
    Eigen::Matrix<double, 3, 1> xyz;
    xyz[0] = dyz[0] * cos(dyz[1]);
    xyz[1] = dyz[0] * sin(dyz[1]);
    xyz[2] = dyz[2];
    return xyz;
}

inline double calc_gimbal_error_dis(ballistic::bullet_res &shoot_ball, Eigen::Vector3d gimbal) {
    return calc_surface_dis_xyz(pyd2xyz(Eigen::Vector3d{gimbal[0], shoot_ball.yaw, gimbal[2]}),
                                pyd2xyz(gimbal));
}

inline double angle_normalize(double x) {
    x -= int(x / (M_PI * 2.0)) * M_PI * 2.0;
    if (x < 0) x += M_PI * 2.0;
    return x;
}

inline void angle_serialize(double &x, double &y) {
    x = angle_normalize(x);
    y = angle_normalize(y);
    if (fabs(x - y) > M_PI) {
        if (x < y)
            x += M_PI * 2.0;
        else
            y += M_PI * 2.0;
    }
}

inline double angle_middle(double x, double y) {
    angle_serialize(x, y);
    return angle_normalize((x + y) / 2);
}

inline double angle_kmiddle(double x, double y, double k) {
    angle_serialize(x, y);
    if (x > y) std::swap(x, y);
    return angle_normalize(x * k + y * (1 - k));
}

inline bool angle_between(double l, double r, double x) {
    angle_serialize(l, r);
    if (l > r) std::swap(l, r);
    angle_serialize(l, x);
    return l <= x && x <= r;
}
#endif