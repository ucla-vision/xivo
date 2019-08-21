// Bundle Adjustment/Pose Graph Optimization module.
// Run in the background, trigger every ... seconds?
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <memory>

#include "core.h"

namespace feh {

class Optimizer;
using OptimizerPtr = Optimizer*;

class Optimizer {
public:
  ~Optimizer();
  static OptimizerPtr Create(const Config &cfg);
  static OptimizerPtr instance();

private:
  Optimizer() = delete;
  Optimizer(const Optimizer &) = delete;
  Optimizer &operator=(const MemoryManager &) = delete;

  Optimizer(const Config &cfg);

  static std::unique_ptr<Optimizer> instance_;
};

} // namespace feh
