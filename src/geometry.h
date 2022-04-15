// Geometry utilities.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include "alias.h"
#include <vector>

namespace xivo {

// Solve the hand-eye calibration problem: AX=XB
// where A, B and X \in SO(3); A & B are known, X is to solve
SO3 HandEyeCalibration(const std::vector<SO3> &A, const std::vector<SO3> &B);
SE3 HandEyeCalibration(const std::vector<SE3> &A, const std::vector<SE3> &B);

// Find the rigid-body-transformation to align two trajectories, i.e., find
// spatial alignment.
// Temporal alignment is assumed.
// Args:
//  Y, X: translational part of the trajectory
// Returns:
//  G \in SE3 such that Y = G X
SE3 TrajectoryAlignment(const std::vector<Vec3> &Y, const std::vector<Vec3> &X);

/** Checks whether or not a vector of points is unique. We assume that the
 *  points in the vector are all different. (This function is used to find
 *  new gauge features. Since features are constrained to be sufficiently far
 *  apart in the tracker by the mask, they are always going to be different.)
 */
bool PointsAreCollinear(const std::vector<Vec3> &pts, number_t thresh);


} // namespace xivo
