#pragma once
/* ********************************* FILE ************************************/
/** \file    p4p.h
 *
 * \brief    This header contains a wrapper for the lambdatwist p3p solver
 *
 * \remark
 * - c++11
 * - no dependencies
 * - can fail, returns identity pose if the data is degenerate, or asserts on incorrect data
 * - no throw
 *
 *
 *
 *
 * \author   Mikael Persson
 * \date     2016-04-01
 * \note MIT licence
 *
 ******************************************************************************/
#include <set>
#include <vector>
#include "utils/cvl/pose.h"


namespace cvl{




/**
 * @brief p4p 3 points for the solution and a 4th to distinguish
 * @param xs, at least 4
 * @param yns, size equal to xs
 * @param indexes, elements in xs, yns
 * @return the Pose which best fits all four.
 * y=project(P*xs)
 *
 * this is the simplest imaginable version of this,
 * the pose will fit the first 3 perfectly and the last if it feels like it.
 * if the measurements are colinear or otherwise degenerate a identity pose is returned.
 *
 * This is suitable for low outlieratios, <50%, and low noise.
 * the use of std::vector is convenient not fast.
 */
PoseD p4p(const std::vector<cvl::Vector3D>& xs,
         const std::vector<cvl::Vector2D>& yns,
          Vector4<uint> indexes);


}// end namespace cvl
