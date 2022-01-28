#pragma once
/** \file    parameters.h
 *
 * \brief    This header contains parameter set for the pnp methods
 *
 * \remark
 *  - header only
 *  - c++11
 *  - no dependencies
 * \todo
 *  - improve structure
 *
 * The purpose of sharing one set of parameters for all pnp ransac variants is to reduce the amount of ifdeffing required to include the ip code
 *
 * \author   Mikael Persson
 * \date     2015-10-01
 * \note MIT licence
 *
 *
 *
 ******************************************************************************/

#include <cmath>


namespace cvl{

/**
 * @brief The PNPParams class
 * common PNP ransac parameters
 */
class PnpParams {
public:

    PnpParams(double threshold=0.001 /*=pixel_threshold/K(0,0), approx 1-3 pixels*/):threshold(threshold){}


    //  parameters
    /// initial inlier ratio estimate, good value = expected inlier ratio/2
    /// should really be estimated online.
    /// double inlier_ratio=0.25;

    /// minimum probability for never findning a single full inlier set
    double min_probability=0.99999; // effectively capped by the max_iterations too...

    /// pixeldist/focallength
    double threshold=0;

    /// perform maximum likelihood sampling consensus,
    //bool MLESAC=false;// not supported at this time... its not better in any of my cases anyways...

    /// break early in the ransac loop if a sufficiently good solution is found
    bool early_exit=false;
    uint early_exit_min_iterations =5;
    double early_exit_inlier_ratio=0.8;

    /// if random eval is used, this is the number of samples drawn
    //uint max_support_eval=500; not used...


    /// a maximum number of iterations to perform to limit the amount of time spent
    uint max_iterations=1000;
    /// the minimum number of iterations to perform to ensure the required number of iterations is correctly computed
    uint min_iterations=100;





    /**
     * @brief get_iterations
     * @param estimated_inliers, start with a conservative guess, say expected_inliers/2.0
     * @param p_inlier_given_noise_and_gt
     * @return
     */
    int get_iterations(double estimated_inliers, double p_meets_treshold_given_inlier_and_gt=0.9){

        double p_inlier=std::min(0.9,estimated_inliers*p_meets_treshold_given_inlier_and_gt);
        p_inlier  = std::min(std::max(p_inlier,1e-2),1-1e-8);
        if(p_inlier<0.01) return max_iterations;

        // this is the range in which the approximation is resonably valid.
        double p_failure = std::min(std::max( 1.0-min_probability,1e-8),0.01);

        // approximate P(inlier|inlier) as 1. note how this differs from P(inlier|inlier,gt)
        double p_good=std::pow(p_inlier,4);
        // approximate hyp as bin
        // approximate bin as norm
        // always draw atleast +50? yeah makes it better
        double iterations=std::ceil((log(p_failure)/log(1.0-p_good))) +50;

        // warning, small number of sample points should increase this further,
        //since bin,norm approxes are bad, this is mostly captured by the +50 and min_iterations, which shouldnt be under 100

        // p_good_given_inlier should be drop with increasing model points too, or noise aswell

        if(iterations<min_iterations) return min_iterations;
        if(iterations>max_iterations) return max_iterations;
        return iterations;


    }


};



}
