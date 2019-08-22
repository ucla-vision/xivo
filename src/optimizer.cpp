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

void Optimizer::AddFeature(FeaturePtr f, const std::vector<Obs> &vobs) {
  // CHECK(!fvertices_.count(f->id()) << "Feature #" << f->id() << " already in optimization graph";
  int fid = f->id();

  if (!fvertices_.count(fid)) {
    // feature vertex not exist, create one
    auto fv = new FeatureVertex();
    fv->setId(fid);
    fv->setMarginalized(true);
    fv->setEstimate(f->Xs());
    fvertices_[fid] = fv;
    optimizer_.addVertex(fv.get());
  }
  auto fv = fvertices_.at(fid);

  for (const auto &obs : vobs) {
    auto g = obs.g;
    int gid = g->id();
    if (!gvertices_.count(gid)) {
      // group vertex not exist, create one
      auto gv = new GroupVertex();
      gv->setId(gid);
      // FIXME (xfei): set pose here, convert to SO3 x R3 to proper type
      gv->setEstimate();
      gv->setAll(); // set aux transforms, may not need this
      // gv->setFixed(true);  // to fix gauge freedom
      optimizer_.addVertex(gv);
    }
    auto gv = gvertices_.at(gid);


    // FIXME (xfei): make sure no duplicate edges are added
    auto e = new Edge ();
    e->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(fv);
    e->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(gv);
    e->setMeasurement(obs.xp);
    e->information() = Mat3::Identity();  // FIXME (xfei): set proper information matrix

    if (use_robust_kernel_) {
      auto rk = new g2o::RobustKernelHuber();
      e->setRobustKernel(rk);
    }
    optimizer_.addEdge(e);
  }



}

} // namespace feh
