#ifndef PERF_HPP_
#define PERF_HPP_
#include <chrono>
#include <string>
#include <unordered_map>
#include <rclcpp/rclcpp.hpp>

class Event {
   private:
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;

   public:
    void start();
    void end();
    double duration();
};

class Perf {
   private:
    double tot = 0, num = 0;
    int threshold;
    rclcpp::Logger logger;

   public:
    Perf(std::string name = "default", int _threshold = 1000);
    double avg();
    void clear();
    void update(double val, int n = 1);
};

class PerfGuard {
   private:
    inline static std::unordered_map<std::string, Perf> map;  // TODO: name in template?
    Event event;
    Perf &perf;

   public:
    PerfGuard(const std::string &);
    ~PerfGuard();
};

#endif