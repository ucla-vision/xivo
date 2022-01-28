#pragma once
/* ********************************* FILE ************************************/
/** \file    klas.pnp.h
 *
 * \brief    This header contains a standard RANSAC pnp solver wrapping the p3p solver by klas.
 *
 *
 * Single refinement of the minimal case solution with a maximum number of inliers or lowest total cutoff error(MLESAC)
 *
 *
 * \remark
 * - c++11
 * - can fail
 * - tested by test_pnp.cpp
 *
 * Dependencies:
 * - ceres solver
 *
 * \todo
 * -losac
 * -pose priors
 *
 *
 *
 * \author   Mikael Persson
 * \date     2016-04-01
 * \note MIT licence
 *
 ******************************************************************************/


#include <utils/cvl/pose.h>
#include <parameters.h>




namespace cvl {
/**
 * @brief pnp_ransac returns Pcw such that X_cam=Pcw*X_world
 * @param xs 3d point in world coordinates
 * @param yns normalized camera measurements
 * @param params, note adjust threshold if neccesary
 * @return
 */
PoseD pnp_ransac(const std::vector<cvl::Vector3D>& xs,
                 const std::vector<cvl::Vector2D>& yns,
                 PnpParams params=PnpParams());


/**
 * @brief The PNP class A basic RANSAC PNP solver
 *
 * Result=est.compute()
 * Pcw=Result.best_pose
 * X_c=Pcw*X_w
 */
class PNP{
public:
    /**
     * @brief PNP constructor given a PnpParam object
     * @param params
     * @param xs 3d points
     * @param yns pinhole normalized 2d measurements
     *
     * copy is cheap, but if this is really the problem, fix later...
     */
    PNP(const std::vector<cvl::Vector3D>& xs,
        const std::vector<cvl::Vector2D>& yns,
        PnpParams params ):xs(xs), yns(yns),params(params){}

    uint total_iters=0;
    /// attempts to compute the solution
    PoseD compute();

protected:




    /// refine a given pose given the stored data, @param pose
    void refine();


    // initialize
    /// the known 3d positions in world coordinates
    std::vector<cvl::Vector3D> xs;
    /// the pinhole normalized measurements corresp to xs
    std::vector<cvl::Vector2D> yns;

    /// parmeter block
    PnpParams params;

    /// number of inliers of best solution
    uint best_inliers=0;



    /// the best pose so far
    cvl::PoseD best_pose;
};

} // end namespace cvl



