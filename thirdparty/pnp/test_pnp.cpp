/* ********************************* FILE ************************************/
/** \file    test_pnp.cpp
 *
 * \brief    This file contains a basic benchmark for pnp solutions.
 *
 * \remark
 * - c++11
 *
 *
 *
 *
 * \author   Mikael Persson
 * \date     2015-04-01
 * \note GPL licence
 *
 ******************************************************************************/


#include <iostream>
#include <utils/mlibtime.h>
#include <pnp_ransac.h>

#include <utils/string_helpers.h>
#include <utils/cvl/pose.h>
#include <simulator.h>







using namespace mlib;
using namespace cvl;
using std::cout;
using std::endl;



class PNP_test{
public:
    virtual PoseD pnp(const std::vector<cvl::Vector3D>& xs,
                      const std::vector<cvl::Vector2D>& yns)=0;
    virtual int getIters()=0;
};


class PNP_lambda: public PNP_test{
public:
    PoseD pnp(const std::vector<cvl::Vector3D>& xs,
              const std::vector<cvl::Vector2D>& yns)
    {
        PnpParams prs;
        cvl::PNP est(xs,yns,prs);

        PoseD pose=est.compute();
        totaliters=est.total_iters;
        return pose;
    }
    int totaliters;
    int getIters(){return totaliters;}
};





void testPnp(PNP_test* pnp){
    std::cerr<<"";
    // generate random poses and pointclouds and test the pnp for them
    std::vector<double> sigmas={0,0.25,0.5,1}; // in pixels

    double outliers=0.5;

    double experiments=1000;

    std::vector<double> failures;failures.resize(sigmas.size(),0);
    std::vector<std::vector<double>> errorss;
    std::vector<std::vector<int>>iterss;

    std::vector<mlib::Timer> timers;timers.resize(sigmas.size());
    for(uint i=0;i<sigmas.size();++i){



        std::vector<double> errors;errors.reserve(experiments);
        std::vector<int> iters;iters.reserve(experiments);
        for(int e=0;e<experiments;++e){
            PointCloudWithNoisyMeasurements data(250,sigmas[i],outliers);

            timers[i].tic();
            PoseD P=pnp->pnp(data.xs,data.yns);
            assert(!std::isnan(P.get4x4().absSum()));
            timers[i].toc();
            iters.push_back(pnp->getIters());


            PoseD I=P*data.Pcw.inverse();
            double error=std::abs(I.getAngle())+I.translation().length();


            errors.push_back(error);

            assert(!std::isnan(error)); // why this works, but not expect true is beyond me...
            //EXPECT_TRUE(!std::isnan(error));
            if((error>0.05)){
                failures[i]++;
            }
        }
        errorss.push_back(errors);
        iterss.push_back(iters);
    }



    std::vector<std::string> headers={"sigma", "bad poses", "ratio","outlier ratio"};



    headers.push_back("mean err");
    headers.push_back("median err");
    headers.push_back("max err");
    //headers.push_back("mean ms");
    headers.push_back("median ms");
    headers.push_back("max ms");
    headers.push_back("median iters");
    std::vector<std::vector<double>> valss;valss.reserve(sigmas.size());

    assert(errorss.size()==sigmas.size());
    for(uint i=0;i<sigmas.size();++i){
        double mean_ms=timers.at(i).getMean().getMilliSeconds();
        //double median_ms=timers.at(i).getMedian().getMilliSeconds();
        double max_ms=timers.at(i).getMax().getMilliSeconds();

        std::vector<double> vals={ sigmas.at(i),failures.at(i),failures.at(i)/experiments,outliers, mean(errorss[i]),median(errorss[i]),max(errorss[i]),mean_ms,max_ms};
        vals.push_back(median(iterss[i]));
        //vals.push_back(max(iterss[i]));

        valss.push_back(vals);
    }
    cout<<"Experiments:   "<<experiments<<endl;
    cout<<"Outlier ratio: "<<outliers<<endl;
    cout<<displayTable(headers,valss)<<endl;
    for(uint i=0;i<sigmas.size();++i)
        assert(experiments<100 || failures.at(i)/experiments<0.05);

}


/*
 * original has the tests for p3p, pose etc aswell, but lets minimize overhead...
TEST(PNP_RANSAC,LAMBDA){

    PNP_lambda pnp;
    testPnp(&pnp);
}
*/


void print_test_for_pybind(){
    PointCloudWithNoisyMeasurements data(10,0,0);
    std::cout<<"xs = np.array([";
    for(int i=0;i<data.xs.size();++i){
        std::cout<<"[";
        for(int j=0;j<3;++j){
            std::cout<< data.xs[i][j];
            if(j!=2)
                std::cout<<", ";
        }
        std::cout<<"]";
        if(i!=data.xs.size()-1)
            std::cout<<",";
    }
    std::cout<<"])\n";

    std::cout<<"ys = np.array([";
    for(int i=0;i<data.yns.size();++i){
        std::cout<<"[";
        for(int j=0;j<2;++j){
            std::cout<< data.yns[i][j];
            if(j!=1)
                std::cout<<", ";
        }
        std::cout<<"]";
        if(i!=data.yns.size()-1)
            std::cout<<",";
    }
    std::cout<<"])\n";

    std::cout<<"pose = np.array([";
    for(int i=0;i<4;++i){


        std::cout<< data.Pcw.q[i];

            std::cout<<", ";
    }

    for(int i=0;i<3;++i){


        std::cout<< data.Pcw.t[i];
        if(i!=2)
            std::cout<<", ";
    }
    std::cout<<"])\n";

    PoseD Pgt(Vector4d(0.898801, -0.0515797, 0.40325, -0.16397).normalized(), Vector3d(0.947568, 0.258716, 0.187565));
    auto M=Pgt.get4x4();
    for(int r=0;r<4;++r){
        for(int c=0;c<4;++c)
            cout<<M(r,c)<<", ";
        cout<<"\n";
    }



}


int main(int argc, char **argv) {
   // print_test_for_pybind();

    PNP_lambda pnp;
    testPnp(&pnp);
    //testing::InitGoogleTest(&argc, argv);    return RUN_ALL_TESTS();

}
