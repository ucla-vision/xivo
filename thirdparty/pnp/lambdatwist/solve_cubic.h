#pragma once


#include <math.h>
#include <iostream>


using std::cout;
using std::endl;

namespace cvl{

/// x*x + b*x +c =>
template<class T> inline bool root2real(T b, T c,T& r1, T& r2){



    T v=b*b -4.0*c;
    if(v<=0)
    {
        r1=r2=-0.5*b;
        return v>=0;
    }

    T y=std::sqrt(v);

    if(b<0){
        r1= 0.5*(-b+y);
        r2= 0.5*(-b-y);
    }else{
        r1= 2.0*c/(-b+y);
        r2= 2.0*c/(-b-y);
    }
    return true;
}


template<class T> constexpr T get_numeric_limit(){ return 1e-13;
    /*
                                            cout<<"ef: "<<std::numeric_limits<float>::max_digits10<<endl;
    cout<<"ed: "<<std::numeric_limits<double>::max_digits10<<endl;
    cout<<"el: "<<std::numeric_limits<long double>::max_digits10<<endl;
*/
}
template<> constexpr float get_numeric_limit<float>(){
    // abs limit is 9 digits
    return 1e-7;
}
template<> constexpr double get_numeric_limit<double>(){
    // abs limit is 17 digits
    return 1e-13;
}
template<> constexpr long double get_numeric_limit<long double>(){
    // abs limit is 21 digits
    return 1e-15 ;
}

#define CountCubic 0

#if CountCubic
std::vector<double> samples;
int tripples=0;
int one_real=0;
int doubles=0;
int three_real=0;
int monotonic=0;
int monotonic0=0;
int monotonic1=0;
int oscillating=0;
int purecomplexpair=0;

#define ADD_CUBIC_COUNT_SAMPLE(sample) samples.reserve(1e7);samples.push_back(sample);
mlib::Timer timer_a,timer_b;

namespace  {
class TossSampleClass{
public:
    ~TossSampleClass(){
        cout<<"Cubic count:   "<<mlib::display(samples)<<endl;
        cout<<"monotonic:   "<<monotonic<<endl;
        cout<<"monotonic0:   "<<monotonic0<<endl;
        cout<<"monotonic1:   "<<monotonic1<<endl;
        cout<<"oscillating: "<<oscillating<<endl;
        cout<<"pure complex pair "<<purecomplexpair<<endl;

        cout<<"one roots: "<<one_real <<endl;
        cout<<"three roots: "<<three_real <<endl;
        cout<<"double roots: "<<doubles <<endl;
        cout<<"tripple roots: "<<tripples<<endl;
        cout<<timer_a<<endl;
        cout<<timer_b<<endl;

    }
} tsc;
}

#else
#define ADD_CUBIC_COUNT_SAMPLE(sample)
#endif
/** 0.5* Number of Newton-Raphson iteratins
 *  ITER is a tuning parameter: incresing it makes the solutions more robust
 * but also more time consuming to compute in a small number of cases
 */
#define KLAS_P3P_CUBIC_SOLVER_ITER 50
#define KLAS_P3P_CUBIC_SOLVER_CHECK 0
/**
 * @brief cubic1  This function finds a single root of the cubic polynomial equation
 * @param b
 * @param c
 * @param d
 * @return
 *
 * h(r) = r^3 + b*r^2 + c*r + d = 0
 *
 * The return root is as stable as possible in the sense that it has as high
 * derivative as possible.  The solution is found by simple Newton-Raphson
 * iterations, and the trick is to choose the intial solution r0 in a clever
 * way.
 *
 * The intial solution is found by considering 5 cases:
 *
 * Cases I and II: h has no stationary points. In this case its derivative
 * is positive.  The inital solution to the NR-iteration is r0 here h has
 * minimal derivative.
 *
 * Case III, IV, and V: has two stationary points, t1 < t2.  In this case,
 * h has negative derivative between t1 and t2.  In these cases, we can make
 * a second order approximation of h around each of t1 and t2, and choose r0
 * as the leftmost or rightmost root of these approximations, depending on
 * whether two, one, or both of h(t1) and h(t2) are > 0.
*/



template<class T> T cubick(T b, T c, T d){

    /* Choose initial solution */
#if 1
    //timer_a.tic();
    T r0;
    // not monotonic
    if (b*b  >= 3.0*c){
        // h has two stationary points, compute them
        //T t1 = t - std::sqrt(diff);
        T v=std::sqrt(b*b -3.0*c);
        T t1 = (-b - v)/(3.0);

        // Check if h(t1) > 0, in this case make a 2-order approx of h around t1
        T k = ((t1+b)*t1+c)*t1+d;

        if (k > 0.0) {
            //Find leftmost root of 0.5*(r0 -t1)^2*(6*t1+2*b) +  k = 0
            r0 = t1 - std::sqrt(-k/(3.0*t1 + b));
            // or use the linear comp too
            //r0=t1 -
        } else {
            T t2 = (-b + v)/(3.0);
            k = ((t2+b)*t2+c)*t2+d;
            //Find rightmost root of 0.5*(r0 -t2)^2*(6*t2+2*b) +  k1 = 0
            r0 = t2 + std::sqrt(-k/(3.0*t2 + b));
        }

        //oscillating++;
    }
    else{
        /*
        r0=1.0/(cubick_inv(c/d,b/d,1.0/d));
        // about half work...
        if(std::abs((((r0+b)*r0+c)*r0+d))>1e-10)
            */
        r0=-b/3.0;
        if(std::abs(((T(3.0)*r0+T(2.0)*b)*r0+c))<1e-4) r0+=1;
        //else r0-=1;
        //T fx=(((r0+b)*r0+c)*r0+d);           r0-=10;            if(fx<0) r0+=20;

    }
    //timer_a.toc();
#endif
    //timer_b.tic();


    //cout<<"base r0: "<<r0<<endl;


    /* Do ITER Newton-Raphson iterations */
    /* Break if position of root changes less than 1e.16 */
    //T starterr=std::abs(r0*(r0*(r0 + b) + c) + d);
    T fx,fpx;

    //cout<<std::setprecision(18)<<get_numeric_limit<T>()<<endl;


    for (unsigned int cnt = 0; cnt < KLAS_P3P_CUBIC_SOLVER_ITER; ++cnt){

        //(+ (* r0 (+  c (* (+ r0 b) r0) )) d )
        fx=(((r0+b)*r0+c)*r0+d);


        if((cnt<7 || std::abs(fx)>get_numeric_limit<T>())  ){
            fpx=((T(3.0)*r0+T(2.0)*b)*r0+c);

            r0-= fx/fpx;
        }
        else
            break;
    }
    //timer_b.toc();

    return r0;
}


}
