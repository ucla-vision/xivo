#pragma once
#include <iostream>

namespace cvl{

template<class T, int iterations>
/**
 * @brief refineL
 * @param L
 * @param a12
 * @param a13
 * @param a23
 * @param b12
 * @param b13
 * @param b23
 *
 * Gauss-Newton Solver
 * For unknown reasons it always works for the correct solution, but not always for the other solutions!
 *
 */
void gauss_newton_refineL(Vector3<T>& L,
                          T a12, T a13, T a23,
                          T b12, T b13, T b23 ){

    // const expr makes it easier for the compiler to unroll
    for(int i=0;i<iterations;++i){
        T l1=L(0);
        T l2=L(1);
        T l3=L(2);
        T r1=l1*l1 + l2*l2 +b12*l1*l2 -a12;
        T r2=l1*l1 + l3*l3 +b13*l1*l3 -a13;
        T r3=l2*l2 + l3*l3 +b23*l2*l3 -a23;

        if(std::abs(r1) +std::abs(r2) +std::abs(r3)<1e-10) break;




        T dr1dl1=(2.0)*l1 +b12*l2;
        T dr1dl2=(2.0)*l2 +b12*l1;
        //T dr1dl3=0;

        T dr2dl1=(2.0)*l1 +b13*l3;
        //T dr2dl2=0;
        T dr2dl3=(2.0)*l3 +b13*l1;


        //T dr3dl1=0;
        T dr3dl2=(2.0)*l2 + b23*l3;
        T dr3dl3=(2.0)*l3 + b23*l2;



        Vector3<T> r(r1, r2, r3);

        // or skip the inverse and make it explicit...
        {

            T v0=dr1dl1;
            T v1=dr1dl2;
            T v3=dr2dl1;
            T v5=dr2dl3;
            T v7=dr3dl2;
            T v8=dr3dl3;
            T det=(1.0)/(- v0*v5*v7 - v1*v3*v8);

            Matrix<T,3,3> Ji( -v5*v7, -v1*v8,  v1*v5,
                              -v3*v8,  v0*v8, -v0*v5,
                              v3*v7, -v0*v7, -v1*v3);
            Vector3<T> L1=Vector3<T>(L) - det*(Ji*r);
            //%l=l - g*H\G;%inv(H)*G
            //L=L - g*J\r; //% works because the size is ok!



            {



                T l1=L1(0);
                T l2=L1(1);
                T l3=L1(2);
                T r11=l1*l1 + l2*l2 +b12*l1*l2 -a12;
                T r12=l1*l1 + l3*l3 +b13*l1*l3 -a13;
                T r13=l2*l2 + l3*l3 +b23*l2*l3 -a23;
                if(std::abs(r11) +std::abs(r12) + std::abs(r13)>std::abs(r1) +std::abs(r2) +std::abs(r3)){
                    // cout<<"bad step: "<< det*(Ji*r)<<++badsteps<<" "<< i<<endl;
                    break;
                }
                else
                    L=L1;
            }
        }




    }
    // cout<<i<<endl;


}


}
