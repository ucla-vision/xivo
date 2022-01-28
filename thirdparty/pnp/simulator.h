#pragma once
#include <utils/cvl/pose.h>
#include <utils/random.h>
namespace cvl{
/// Uniform distribution on the unit sphere
template<class T,int R> cvl::Vector<T,R> getRandomUnitVector(){
    static_assert(R>1, "not much of a vector otherwize... " );
    cvl::Vector<T,R> n;

    for(int i =0;i<R;++i)
        n[i]=mlib::randn<T>(0,1);

    // can happen...
    if(n.abs().sum() <1e-10)
        return getRandomUnitVector<T,R>();

    n.normalize();
    return n;
}

template<class T> Matrix<T,3,3> getRandomRotation(){
    cvl::Vector4<T> q=getRandomUnitVector<double,4>();
    return getRotationMatrix(q);
}




std::vector<cvl::Vector3D> getRandomPointsInfrontOfCamera(cvl::PoseD Pcw,
                                                          uint N) {
    // reduce the odds that it just cycles!
    std::vector<cvl::Vector3D> xs;xs.reserve(N);
    PoseD Pwc=Pcw.inverse();
    while(xs.size()<N){
        cvl::Vector2D yn=cvl::Vector2D(mlib::randu<double>(-1,1),mlib::randu<double>(-1,1));
        double distance =mlib::randu<double>(0.1,100); // about 0.1 to 1000 m(if 0,7)
        Vector3D xc=(yn.homogeneous()*distance);
        xs.push_back(Pwc*xc);
    }
    assert(xs.size()==N);
    return xs;
}



class PointCloudWithNoisyMeasurements{
public:
    PointCloudWithNoisyMeasurements(uint N,double pixel_sigma,double outlier_ratio){
            // generate 500 points infront of the camera

            Pcw=cvl::PoseD(getRandomRotation<double>(),getRandomUnitVector<double,3>());
            xs=getRandomPointsInfrontOfCamera(Pcw,N);
            yns_gt.reserve(xs.size());
            yns.reserve(yns_gt.size());

            for(auto x:xs){
                auto y=(Pcw*x).dehom();
                yns_gt.push_back(y);
            }



            double sigma=pixel_sigma*0.001;

            for(cvl::Vector2D y:yns_gt){
                yns.push_back(y+getRandomUnitVector<double,2>()*sigma);
            }


            if(outlier_ratio>0){
                int outliers=xs.size()*outlier_ratio;
                for(int i=0;i<outliers;++i){
                    // exact ratio is not needed
                    int index=mlib::randui<int>(0,yns.size()-1);

                    auto y=yns[index];
                    if(mlib::randu<double>(0,1)>0.5) // and something to really mess with things that dont cut
                        y+=getRandomUnitVector<double,2>();
                    while(((Pcw*xs[index]).dehom() - y).norm() < 0.002 + pixel_sigma*0.001*3)
                        y+=getRandomUnitVector<double,2>()*0.1*mlib::randu<double>(3,10);
                    yns.at(index)=y; // near but not close enough for a missleading match
                }
            }




        }


    std::vector<cvl::Vector3d> xs;
    std::vector<cvl::Vector2d> yns,yns_gt;
    cvl::PoseD Pcw;
};

}// end namespace cvl
