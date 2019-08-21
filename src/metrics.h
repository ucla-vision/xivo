// Evaluation metrics, namely, ATE and RTE for trajectory evaluation.
// And surface error for reconstruction/object alignment, etc.
// Reference:
// David Schubert, Thore Goll, Nikolaus Demmel, Vladyslav Usenko, Jorg Stuckler
// and Daniel Cremers
// The TUM VI Benchmark for Evaluating Visual-Inertial Odometry
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <chrono>
#include <tuple>
#include <vector>

#include "message_types.h"

namespace feh {

// Compute Absolute Trajectory Error given estimated trajectory and ground truth
// Args:
//  est, gt: Estimated (Groundtruth) trajectories
//  resolution: if estimate exactly at t+dt cannot be found, return the closest
//  one within +/- resolution seconds
// Returns: ATE
std::tuple<number_t, SE3> ComputeATE(const std::vector<msg::Pose> &est,
                                  const std::vector<msg::Pose> &gt,
                                  number_t res = 0.005);
// Compute Relative Pose Error given estimated trajectory and ground truth
// Args:
//  est, gt: Estimated (Groundtruth) trajectories
//  dt: time interval to compute error
//  resolution: if estimate exactly at t+dt cannot be found, return the closest
//  one within +/- resolution seconds
// Returns: RPE
std::tuple<number_t, number_t> ComputeRPE(const std::vector<msg::Pose> &est,
                                    const std::vector<msg::Pose> &gt,
                                    number_t dt = 1.0, number_t res = 0.005);

} // namespace feh
