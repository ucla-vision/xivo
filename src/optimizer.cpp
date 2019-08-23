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

Optimizer::~Optimizer() {
}

Optimizer::Optimizer(const Json::Value &cfg) 
  : verbose_{false}, solver_type_{"cholmod"}, use_robust_kernel_{false}, initialized_{false}

{
  // setup flags
  verbose_ = cfg.get("verbose", true).asBool();
  solver_type_ = cfg.get("solver", "cholmod").asString();
  use_robust_kernel_ = cfg.get("use_robust_kernel", true).asBool();

  if (solver_type_ == "cholmod") {
    solver_ = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
  } else if (solver_type_ == "csparse") {
    solver_ = g2o::make_unique<g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>>();
  } else if (solver_type_ == "dense") {
    solver_ = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
  }

  optimizer_.setVerbose(verbose_);

  algorithm_ = std::make_unique<g2o::OptimizationAlgorithmLevenberg>(
      g2o::make_unique<g2o::BlockSolver_6_3>(std::move(solver_)));

  optimizer_.setAlgorithm(algorithm_.get());

}


FeatureVertex* Optimizer::CreateFeatureVertex(const FeatureAdapter &f) {
  auto fv = new FeatureVertex();
  fv->setId(f.id);
  fv->setMarginalized(true);
  fv->setEstimate(f.Xs);
  fvertices_[f.id] = fv;
  optimizer_.addVertex(fv);
  return fv;
}

GroupVertex* Optimizer::CreateGroupVertex(const GroupAdapter &g) {
  auto gv = new GroupVertex();
  gv->setId(g.id);
  gv->setEstimate(g.gsb);
  // FIXME (xfei): to fix gauge freedom
  // gv->setFixed(true);  
  optimizer_.addVertex(gv);
  return gv;
}

Edge* Optimizer::CreateEdge(FeatureVertex *fv, GroupVertex *gv, const Vec2 &xp, const Mat2 &IM) {
  auto e = new Edge();
  e->setVertex(0, fv);
  e->setVertex(1, gv);
  e->setMeasurement(xp);
  if (IM.isZero(0)) {
    e->setInformation(Mat2::Identity());
  } else {
    e->setInformation(IM);  // IM = Information Matrix
  }

  if (use_robust_kernel_) {
    auto rk = new g2o::RobustKernelHuber();
    e->setRobustKernel(rk);
  }
  optimizer_.addEdge(e);
  return e;
}

void Optimizer::AddFeature(const FeatureAdapter &f, const std::vector<ObsAdapterG> &obs) {
  // CHECK(!fvertices_.count(f->id()) << "Feature #" << f->id() << " already in optimization graph";

  if (!fvertices_.count(f.id)) {
    // feature vertex not exist, create one
    CreateFeatureVertex(f);
  }
  auto fv = fvertices_.at(f.id);

  for (auto [g, xp, IM]: obs) {
    if (!gvertices_.count(g.id)) {
      // group vertex not exist, create one
      CreateGroupVertex(g);
    }
    auto gv = gvertices_.at(g.id);
    // FIXME (xfei): make sure no duplicate edges are added
    CreateEdge(fv, gv, xp, IM);
  }
}

void Optimizer::AddGroup(const GroupAdapter &g, const std::vector<ObsAdapterF> &obs) {
  if (!gvertices_.count(g.id)) {
    CreateGroupVertex(g);
  }
  auto gv = gvertices_.at(g.id);

  for (auto [f, xp, IM]: obs) {
    if (!fvertices_.count(f.id)) {
      CreateFeatureVertex(f);
    }
    auto fv = fvertices_.at(f.id);
    CreateEdge(fv, gv, xp, IM);
  }

}

void Optimizer::Solve(int max_iters) {
  if (!initialized_) {
    optimizer_.initializeOptimization();
    initialized_ = true;
  }

  optimizer_.setVerbose(verbose_);
  optimizer_.optimize(max_iters);
}


} // namespace feh
