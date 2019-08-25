// light-weight header only tinmer
// Author: Xiaohan Fei
#pragma once

#include <ostream>
#include <chrono>
#include <memory>

namespace xivo {

/// \brief timer
class Timer {
public:
  struct Event {
    std::chrono::high_resolution_clock::time_point latest;  // latest start time
    std::chrono::nanoseconds duration; // total duration
    int occurrence;  // how many 
  };
public:

  friend std::ostream &operator<<(std::ostream &os, const Timer& t) {
    os << "....." << std::endl;
    for (const auto &p : t.data_) {
      const auto &e{p.second};
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(e.duration);
      os << "[" << t.name_ << "]" 
        << p.first 
        << ":" << ms.count() / (float)e.occurrence << " ms\n";
    }
    return os;
  }

  Timer(const std::string &name = "default")
      : name_{name} {}

  void Tick(const std::string &event) {
    data_[event].latest = std::chrono::high_resolution_clock::now();
  }

  auto Tock(const std::string &event) -> std::chrono::milliseconds {
    auto duration = SingleOccurrenceDuration(event);

    if (data_.count(event)) {
      Event& e = data_[event];
      e.duration += duration; 
      e.occurrence += 1;
    } else {
      Event& e = data_[event];
      e.duration = duration; 
      e.occurrence = 1;
    }
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration);
  }

  void Reset() {
    data_.clear();
  }
  virtual ~Timer() = default;

protected:
  std::chrono::nanoseconds SingleOccurrenceDuration(const std::string &event) const {
    auto tmp = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        tmp - data_.at(event).latest);
  }

  std::unordered_map<std::string, Event> data_;
  std::string name_;
};

class GlobalTimer: public Timer {
public:
  static GlobalTimer* instance() {
    if (!instance_) {
      instance_ = std::unique_ptr<GlobalTimer>();
    }
    return instance_.get();
  }

private:
  GlobalTimer(): Timer{"global"} {}
  static std::unique_ptr<GlobalTimer> instance_;
};

}


