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
  if (auto solver_type{cfg.get("solver", "cholmod").asString()};
      solver_type == "cholmod") {
    // FIXME (xfei): what is PoseMatrixType? and what does _6_3 mean?
    solver_ = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
  } else if (solver_type == "csparse") {
    solver_ = g2o::make_unique<g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>>();
  } else if (solver_type == "dense") {
    solver_ = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
  }

  optimizer_.setVerbose(cfg.get("verbose", true).asBool());

  algorithm_ = std::make_unique<g2o::OptimizationAlgorithmLevenberg>(
      g2o::make_unique<g2o::BlockSolver_6_3>(std::move(solver_)));

  optimizer_.setAlgorithm(algorithm_.get());

}

} // namespace feh
