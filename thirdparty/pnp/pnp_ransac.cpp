#include <iostream>
#include <ceres/problem.h>
#include <ceres/jet.h>
#include <ceres/loss_function.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/local_parameterization.h>
#include <ceres/solver.h>


#include <utils/mlibtime.h>
#include <utils/random.h>
#include <pnp_ransac.h>
#include <p4p.h>



using std::cout;using std::endl;


namespace cvl {


PoseD pnp_ransac(const std::vector<cvl::Vector3D>& xs,
                 const std::vector<cvl::Vector2D>& yns,
                 PnpParams params){
    PNP pnp(xs,yns,params);
    return pnp.compute();
}

/**
 * @brief evaluate_inlier_set
 * @param xs
 * @param yns
 * @param threshold
 * @param pose
 * @param best_inliers
 * @return number of inliers if greater than best_inliers, otherwize slightly less...
 * set best_inliers to 0 to compute the exact count...
 *
 */
uint evaluate_inlier_set(const std::vector<cvl::Vector3D>& xs,
                        const std::vector<cvl::Vector2D>& yns,
                        double threshold,
                        PoseD pose,
                        uint best_inliers){
    // this is the by far slowest part of the system!



    uint inliers=0;


    Matrix4x4D M=pose.get4x4(); // much faster...


    double threshold_squared=threshold*threshold; // we are comparing to the square of it after all...

    for(uint i=0;i<xs.size();++i){

        //cout<<((pose*xs[i]).dehom() - yns[i]).squaredNorm()<<"   "<<threshold_squared<<endl;

        cvl::Vector4D X=xs[i].homogeneous(); // yes even with the extra cost here...
//        Vector3D XR=(M*X).dehom(); // technically based on how 4x4 etc work, no dehom required
        Vector4D XR=(M*X);

        double x=XR[0];
        double y=XR[1];
        // any negative value is behind the camera, those are outliers by any definition!
        // however very distant points behave fairly well... even with the abs... so break is needed
        double iz=1.0/XR[2];

        if(iz<0) continue;


        double err1=x *iz - yns[i](0);
        double err2=y *iz - yns[i](1);

        double err=err1*err1 + err2*err2;

        inliers += (err < threshold_squared) ? 1 : 0;
        //mle += std::min(errors[i],thr);// use this to compute mle instead...
        // highest number of inliers possible at this point. inliers + (xs.size()) -i
        if(((xs.size()-i +inliers)<best_inliers)) break;
    }
    return inliers;

}

/**
 * @brief The PnPReprojectionError class pnp cost for ceres
 *
 * Note:
 * - The cost is will become vectorized if compiled with optimization
 *
 */
class PnPReprojectionError
{


public:
    /**
     * @brief PnPReprojectionError
     * @param xs
     * @param yns
     */
    PnPReprojectionError(const std::vector<Vector3D>& xs,
                         const std::vector<Vector2D>& yns)  {
        this->xs=xs;
        this->yns=yns;
    }

    template <typename T>
    /**
     * @brief operator () autodiff enabled error
     * @param rotation
     * @param translation
     * @param residuals
     * @return
     */
    bool operator()(const T* const rotation, const T* const translation,T* residuals) const
    {
        // Get camera rotation and translation
        cvl::Pose<T> P(rotation,translation);
        //cvl::Matrix3x3<T> R=P.getR();
        cvl::Matrix4x4<T> M=P.get4x4(); // surprisingly faster
        //cvl::Vector3<T> tr(translation,true);
        for (uint i = 0; i < xs.size(); ++i) {

            cvl::Vector4<T> x=xs[i].homogeneous();
            cvl::Vector4<T> xr=M*x;  //known 1 at the end
            T iz=T(1.0/xr[2]);


            residuals[0]   = xr[0] *iz - T(yns[i][0]);
            residuals[1] = xr[1] *iz - T(yns[i][1]);
            residuals+=2;
        }
        return true;
    }
    /// the 3d point observations
    std::vector<Vector3D> xs;
    /// the pinhole normalized image observations
    std::vector<Vector2D> yns;
    /**
     * @brief Create Autodiff error factory
     * @param inlier_xs
     * @param inlier_yns
     * @return
     */
    static ceres::CostFunction* Create(const std::vector<Vector3D>& inlier_xs,
                                       const std::vector<Vector2D>& inlier_yns ){
        return new ceres::AutoDiffCostFunction<PnPReprojectionError, ceres::DYNAMIC, 4,3>(
                    new PnPReprojectionError(inlier_xs,inlier_yns), inlier_xs.size()*2);
    }

};




Vector4<uint> get4RandomInRange0(uint max){
    Vector4<uint> indexes;
    //todo verify the fast one!
    /*
    n=0;
    while(n<3){
        uint val=randui<int>(0,max-1);
        for(int i=0;i<n;++i)
            if(indexes[i]==val) continue;
        indexes[n++]=val;
    }*/

    // for large numbers in smallish sets, sorting is faster
    std::set<uint> set;
    assert(4<=max);
    while(set.size()<4)
        set.insert(mlib::randui<int>(0,max-1));
    int n=0;
    for(uint i:set){
        indexes[n++]=i;
    }
    return indexes;
}




PoseD PNP::compute(){
    double inlier_estimate=best_inliers/((double)xs.size());
    uint iters=params.get_iterations(inlier_estimate);

    uint i;
    for(i=0;i<iters;++i){
        // pick 4 at random,

        // will always succeed, returns identity on degeneracy...
        PoseD pose=p4p(xs,yns,get4RandomInRange0(xs.size()));
        assert(pose.isnormal());
        if(!pose.isnormal()) continue;
        //cout<<pose<<endl;

        // evaluate inlier set
        // timer.tic();

        uint inliers=evaluate_inlier_set(xs,yns,params.threshold,pose,best_inliers);

        // timer.toc();

        if(inliers>best_inliers){
            //std::cout<<"inliers: "<<inliers<<std::endl;
            best_inliers=inliers;
            best_pose=pose;

            // recompute only when neccessary its expensive...
            double inlier_estimate=best_inliers/((double)xs.size());
            iters=params.get_iterations(inlier_estimate);

            // perform early exit if exit criteria met
            if( false &&    params.early_exit &&
                    i>params.early_exit_min_iterations &&
                    best_inliers>params.early_exit_inlier_ratio*xs.size()
                    ) break;
        }
    }

    total_iters=i;


    // refine pose, if possible...
    if(best_inliers>3)        refine();
    return best_pose; // will be identity if a complete failure...
}



/**
 * @brief PNP::refine
 * since we expect a high noise, low outlier ratio solution(<50%), we should refine using a cutoff loss twice...
 */
void PNP::refine(){

    std::vector<Vector3D> inlier_xs;inlier_xs.reserve(xs.size());
    std::vector<Vector2D> inlier_yns;inlier_yns.reserve(xs.size());
    std::vector<int> inliers; inliers.resize(xs.size(),0);
    double thr=params.threshold*params.threshold;

    {
        // think about if you can use an explicit cutoff loss here...
        // tests show its faster to separate the inliers and call it twice though...


        inlier_xs.clear();
        inlier_yns.clear();
        for(uint i=0;i<xs.size();++i){
            Vector3d xr=best_pose*xs[i];
            if(xr[2]<0) continue;
            if(((xr).dehom() - yns[i]).squaredNorm()>thr) continue;
            inlier_xs.push_back(xs[i]);
            inlier_yns.push_back(yns[i]);
            inliers[i]=1;
        }

        ceres::Problem problem;
        ceres::LossFunction* loss=nullptr;// implies squared
        problem.AddResidualBlock(PnPReprojectionError::Create(inlier_xs,inlier_yns),loss,best_pose.getRRef(),best_pose.getTRef());


        ceres::LocalParameterization* qp = new ceres::QuaternionParameterization;
        problem.SetParameterization(best_pose.getRRef(), qp);

        ceres::Solver::Options options;{
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.max_num_iterations=5;
            options.function_tolerance=1e-6;     // default 1e-6
            options.gradient_tolerance=1e-6;     //default 1e-4*function_tolerance
        }
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        //out<<"Report 1: \n"<<summary.FullReport()<<endl;
    }
    // check somehow if there is a big difference in the inliers?


    {
        // think about if you can use an explicit cutoff loss here...
        // tests show its faster to separate the inliers and call it twice though...


        inlier_xs.clear();
        inlier_yns.clear();
        int deltas=0;
        for(uint i=0;i<xs.size();++i){
            Vector3d xr=best_pose*xs[i];
            bool inlier=true;
            if(xr[2]<0) inlier=false;
            if(((xr).dehom() - yns[i]).squaredNorm()>thr) inlier=false;
            if(inlier ^ (inliers[i]==1)) deltas++;
            if(!inlier) continue;
            inlier_xs.push_back(xs[i]);
            inlier_yns.push_back(yns[i]);
        }
        // if to few changes hav occured, dont do a second refine...
        if(deltas<0.05*inlier_xs.size()) return;
        ceres::Problem problem;

        ceres::LossFunction* loss=nullptr;//


        problem.AddResidualBlock(PnPReprojectionError::Create(inlier_xs,inlier_yns),loss,best_pose.getRRef(),best_pose.getTRef());


        ceres::LocalParameterization* qp = new ceres::QuaternionParameterization;
        problem.SetParameterization(best_pose.getRRef(), qp);

        ceres::Solver::Options options;{
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.max_num_iterations=3;
            options.function_tolerance=1e-8;     // default 1e-6
            options.gradient_tolerance=1e-8;     //default 1e-4*function_tolerance
        }
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        //cout<<"Report 1: \n"<<summary.FullReport()<<endl;
    }

}



} // end namespace cvl

