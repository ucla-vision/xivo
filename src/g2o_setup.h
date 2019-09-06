// g2o headers all-in-one.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once

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
