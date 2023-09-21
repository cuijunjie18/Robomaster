#include <rm_utils/perf.hpp>

void Event::start() { start_time = std::chrono::steady_clock::now(); }

void Event::end() { end_time = std::chrono::steady_clock::now(); }

double Event::duration() {
    return std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

Perf::Perf(std::string name, int _threshold) : logger(rclcpp::get_logger(name)) {
    threshold = _threshold;
}

void Perf::update(double val, int n) {
    tot += val;
    num += n;
    if (num > threshold) {
        // logger.sinfo(
        //     "[PERF] time: {:.3f}ms fps: {:.2f}",
        //     tot / num,
        //     num / tot * 1000
        // );
        RCLCPP_INFO(logger, "[PERF] time: %.3fms fps: %.2f", tot / num, num / tot * 1000);
        clear();
    }
}

void Perf::clear() { tot = num = 0; }

double Perf::avg() { return tot / num; }

PerfGuard::PerfGuard(const std::string& _name)
    : perf([&_name] {
          if (auto it = map.find(_name); it != map.end()) {
              return std::ref(it->second);
          } else {
              map.insert(std::make_pair(_name, Perf(_name)));
              return std::ref(map[_name]);
          }
      }()) {
    event.start();
}

PerfGuard::~PerfGuard() {
    event.end();
    perf.update(event.duration());
}
