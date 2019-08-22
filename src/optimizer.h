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

namespace feh {

class Optimizer;
using OptimizerPtr = Optimizer*;


class Optimizer {
public:
  using FeatureVertex = g2o::VertexSBAPointXYZ;
  using GroupVertex = g2o::VertexCam;
  using Edge = g2o::EdgeProjectP2MC;

  struct FeatureAdapter {
    int id;
    Vec3 Xs;  // 3D coordinates in spatial frame
  };

  struct GroupAdapter {
    int id;
    SE3 gsb;  // body to spatial transformation
  };

  using ObsAdapter = std::tuple<GroupAdapter, Vec2>;

public:
  ~Optimizer();
  static OptimizerPtr Create(const Json::Value &cfg);
  static OptimizerPtr instance();
  void Solve(int iters=1);

  void AddFeature(const FeatureAdapter &f, const std::vector<ObsAdapter> &obs);
  void AddGroup(const GroupAdapter &g, const std::vector<FeatureAdapter> &obs);

private:
  Optimizer() = delete;
  Optimizer(const Optimizer &) = delete;
  Optimizer &operator=(const Optimizer &) = delete;

  Optimizer(const Json::Value &cfg);

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
