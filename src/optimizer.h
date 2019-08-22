// Bundle Adjustment/Pose Graph Optimization module.
// Run in the background, trigger every ... seconds?
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <memory>
#include <tuple>

// g2o
#include "g2o/config.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/icp/types_icp.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"

#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"

#include "json/json.h"

#include "alias.h"
#include "group.h"

namespace feh {

class GroupVertex: public g2o::BaseVertex<6, SE3> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GroupVertex();

  virtual void setToOriginImpl() {
    _estimate = SE3{};  // gsb: body-to-spatial
  }

  virtual void oplusImpl(double* update) {
    _estimate.R() *= SO3::exp(Eigen::Map<Vec3>(update));
    _estimate.T() += Eigen::Map<Vec3>(update+3);
  }

  virtual bool read(std::istream& is);
  virtual bool write(std::ostream& os) const;
};

class FeatureVertex: public <3, Vec3> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FeatureVertex();

  virtual void setToOriginImpl() {
  }

  virtual void oplusImpl(double* update) {
    for (int i = 0; i < 3; ++i) {
      _estimate[i] += update[i];
    }
  }

  virtual bool read(std::istream& is);
  virtual bool write(std::ostream& os) const;
};

class Edge: public BaseBinaryEdge<2, Vec2,  FeatureVertex, GroupVertex> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Edge();

  void computeError() {
    const FeatureVertex* fv = static_cast<const FeatureVertex*>(_vertices[0]);
    const GroupVertex* gv = static_cast<const GroupVertex*>(_vertices[1]);
    // gsb.inv -> gbs
    Vec3 Xb = gv->estimate().inv() * fv->estimate();
    _error = Xb.head<2>() / Xb(2);
  }

  virtual bool read(std::istream& is);
  virtual bool write(std::ostream& os) const;
};

struct FeatureAdapter {
  int id;
  Vec3 Xs;  // 3D coordinates in spatial frame
};

struct GroupAdapter {
  int id;
  SE3 gsb;  // body to spatial transformation
};

struct ObsAdapterG: std::tuple<GroupAdapter, Vec2, Mat2>;
struct ObsAdapterF: std::tuple<FeatureAdapter, Vec2, Mat2>;


class Optimizer;
using OptimizerPtr = Optimizer*;


class Optimizer {
public:


public:
  ~Optimizer();
  static OptimizerPtr Create(const Json::Value &cfg);
  static OptimizerPtr instance();
  void Solve(int iters=1);

  void AddFeature(const FeatureAdapter &f, const std::vector<ObsAdapterG> &obs);
  void AddGroup(const GroupAdapter &g, const std::vector<ObsAdapterF> &obs);

private:
  Optimizer() = delete;
  Optimizer(const Optimizer &) = delete;
  Optimizer &operator=(const Optimizer &) = delete;

  Optimizer(const Json::Value &cfg);

  FeatureVertex* CreateFeatureVertex(const FeatureAdapter &f);
  GroupVertex* CreateGroupVertex(const GroupAdapter &g);
  Edge* CreateEdge(FeatureVertex *fv, GroupVertex *gv, const Vec2 &xp, const Mat2 &IM);

  // the instance class memeber
  static std::unique_ptr<Optimizer> instance_;

  // flags
  
  // graph structure: features & groups as vertices
  // std::unordered_map<int, std::unique_ptr<FeatureVertex>> fvertices_;
  // std::unordered_map<int, std::unique_ptr<GroupVertex>> gvertices_;
  std::unordered_map<int, FeatureVertex*> fvertices_;
  std::unordered_map<int, GroupVertex*> gvertices_;

  // g2o variables
  g2o::SparseOptimizer optimizer_;
  std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> solver_;
  std::unique_ptr<g2o::OptimizationAlgorithmLevenberg> algorithm_;
};

} // namespace feh
