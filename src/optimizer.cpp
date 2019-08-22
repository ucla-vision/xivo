#include <iostream>

// 3rdparty
#include "glog/logging.h"

// feh
#include "optimizer.h"

namespace feh {

std::unique_ptr<Optimizer> Optimizer::instance_ = nullptr;

OptimizerPtr Optimizer::Create(const Json::Value &cfg) {
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

Optimizer::Optimizer(const Json::Value &cfg) {
  if (auto solver_type{cfg.get("solver", "cholmod").asString()};
      solver_type == "cholmod") {
    // _6_3: poses are parametrized by 6-dim vectors and landmarks by 3-dim vectors
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

void Optimizer::AddFeature(const FeatureAdapter &f, const std::vector<ObsAdapter> &obs) {
  // CHECK(!fvertices_.count(f->id()) << "Feature #" << f->id() << " already in optimization graph";

  if (!fvertices_.count(f.id)) {
    // feature vertex not exist, create one
    auto fv = new FeatureVertex();
    fv->setId(f.id);
    fv->setMarginalized(true);
    fv->setEstimate(f.Xs);
    fvertices_[f.id] = fv;
  }
  auto fv = fvertices_.at(f.id);

  /*
  for (auto [g, xp]: obs) {
    if (!gvertices_.count(g.id)) {
      // group vertex not exist, create one
      auto gv = new GroupVertex();
      gv->setId(g.id);
      // FIXME (xfei): set pose here, convert to SO3 x R3 to proper type
      gv->setEstimate();
      gv->setAll(); // set aux transforms, may not need this
      // gv->setFixed(true);  // to fix gauge freedom
      optimizer_.addVertex(gv);
    }
    auto gv = gvertices_.at(g.id);

    // FIXME (xfei): make sure no duplicate edges are added
    auto e = new Edge ();
    e->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(fv);
    e->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(gv);
    e->setMeasurement(xp);
    e->information() = Mat3::Identity();  // FIXME (xfei): set proper information matrix

    if (use_robust_kernel_) {
      auto rk = new g2o::RobustKernelHuber();
      e->setRobustKernel(rk);
    }
    optimizer_.addEdge(e);
  }
  */
}


} // namespace feh
