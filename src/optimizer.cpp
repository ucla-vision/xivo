#include <iostream>

// 3rdparty
#include "glog/logging.h"

// feh
#include "optimizer.h"

namespace feh {

std::unique_ptr<Optimizer> Optimizer::instance_ = nullptr;

OptimizerPtr Optimizer::Create(const Config &cfg) {
  if (instance_) {
    LOG(WARNING) << 
      "Optimizer instance already created! Returning existing one ...";
  } else {
    instance_ = std::unique_ptr<Optimizer>(new Optimizer(cfg));
  }
  return instance_.get();
} 

OptimizerPtr Optimizer::instance() {
  return instance_.get();
}

Optimizer::Optimizer(const Config &cfg) {
}

} // namespace feh
