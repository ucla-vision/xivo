#pragma once

#include <lambdatwist/solve_cubic.h>
#include <lambdatwist/solve_eig0.h>
#include <lambdatwist/refine_lambda.h>
#include <lambdatwist/p3p_timers.h>



using std::endl;using std::cout;












namespace cvl{









template<class T, int refinement_iterations=5>
int p3p_lambdatwist( Vector3<T> y1,
                     Vector3<T> y2,
                     Vector3<T> y3,
                     Vector3<T> x1,
                     Vector3<T> x2,
                     Vector3<T> x3,
                     Vector<cvl::Matrix<T,3,3>,4>& Rs,
                     Vector<Vector3<T>,4>& Ts){





    // normalize the length of ys
    TIC(lt1);

    //Vector3<T> y1=ys_in[0];
    //Vector3<T> y2=ys_in[1];
    //VecWith a mean decrease from $490ns$ to $260ns$ for repeated calls the system performs very nearly twice as fast. The variance and maximums are also lower which is important in real time systems.tor3<T> y3=ys_in[2];
    // takes about 8ns

    y1.normalize();
    y2.normalize();
    y3.normalize();


    T b12=-2.0*(y1.dot(y2));
    T b13=-2.0*(y1.dot(y3));
    T b23=-2.0*(y2.dot(y3));




    // implicit creation of Vector3<T> can be removed
    Vector3<T> d12=x1-x2;
    Vector3<T> d13=x1-x3;
    Vector3<T> d23=x2-x3;
    Vector3<T> d12xd13(d12.cross(d13));



    T a12=d12.squaredLength();
    T a13=d13.squaredLength();
    T a23=d23.squaredLength();

    //if(abs(D1.determinant())<1e-5 || fabs(D2.determinant())<1e-5)        cout<<"det(D): "<<D1.determinant()<<" "<<D2.determinant()<<endl;
    TOC(lt1);


    TIC(lt2);



    //a*g^3 + b*g^2 + c*g + d = 0
    T c31=-0.5*b13;
    T c23=-0.5*b23;
    T c12=-0.5*b12;
    T blob=(c12*c23*c31 - 1.0);

    T s31_squared=1.0-c31*c31;
    T s23_squared=1.0-c23*c23;
    T s12_squared=1.0-c12*c12;



    T p3 = (a13*(a23*s31_squared - a13*s23_squared));

    T p2 = 2.0*blob*a23*a13 + a13*(2.0*a12 + a13)*s23_squared + a23*(a23 -     a12)*s31_squared;

    T p1 = a23*(a13 - a23)*s12_squared - a12*a12*s23_squared - 2.0*a12*(blob*a23 +    a13*s23_squared);

    T p0 = a12*(a12*s23_squared - a23*s12_squared);

    TOC(lt2);
    TIC(lt3);
    T g=0;

    //p3 is det(D2) so its definietly >0 or its a degenerate case
    /* the root with the highest gradient magnitude should be selected,
     * thats built into the cubick initialization.*/
    {
        p3=1.0/p3;
        p2*=p3;
        p1*=p3;
        p0*=p3;

        // get sharpest real root of above...

        g=cubick(p2,p1,p0);
    }

    //  cout<<"g: "<<g<<endl;


    TOC(lt3);
    // we can swap D1,D2 and the coeffs!
    // oki, Ds are:
    //D1=M12*XtX(2,2) - M23*XtX(1,1);
    //D2=M23*XtX(3,3) - M13*XtX(2,2);

    //[    a23 - a23*g,                 (a23*b12)/2,              -(a23*b13*g)/2]
    //[    (a23*b12)/2,           a23 - a12 + a13*g, (a13*b23*g)/2 - (a12*b23)/2]
    //[ -(a23*b13*g)/2, (a13*b23*g)/2 - (a12*b23)/2,         g*(a13 - a23) - a12]
    TIC(lt4);

    // gain 13 ns...
    T A00=        a23*(1.0- g);
    T A01=         (a23*b12)*0.5;
    T A02=        (a23*b13*g)*(-0.5);
    T A11=    a23 - a12 + a13*g;
    T A12=        b23*(a13*g - a12)*0.5;
    T A22=  g*(a13 - a23) - a12;



    Matrix<T,3,3> A(A00,A01,A02,
                    A01,A11,A12,
                    A02,A12,A22);




    // get sorted eigenvalues and eigenvectors given that one should be zero...
    Matrix<T,3,3> V;
    Vector3<T> L;



    eigwithknown0(A,V,L);

    T v=std::sqrt(std::max(T(0),-L(1)/L(0)));
    //cout<<"V "<<V<<endl;
    //cout<<"L "<<L<<endl;

    TOC(lt4);
    TIC(lt5);


    int valid=0;
    Vector<Vector<T,3>,4> Ls;
    Vector2<T> ss(v,-v);
    for(T s:ss)
    {

        // u = V(:, 1) - sV(:,2)

        T u1=V(0) - s*V(1);
        T u2=V(3) - s*V(4);
        T u3=V(6) - s*V(7);

        // we are computing lambda using a linear relation
        // u1*l1 + u2*l2 + u3*l3=0
        // li>0, implies all three ui=0 is degenerate...
        // if two are zero the third must be
        // hence at most one can be zero.
        // divide by the largest for best numerics,
        // simple version, use the bigger of u1, u2, one will always be non-zero
        if(std::abs(u1)<std::abs(u2))
        {
            // solve for l2
            T a= (a23 - a12)*u3*u3 - a12*u2*u2 + a12*b23*u2*u3;
            T b= (T(2)*a23*u1*u3 - T(2)*a12*u1*u3 + a12*b23*u1*u2 - a23*b12*u2*u3)/a;
            T c= (a23*u1*u1 - a12*u1*u1 + a23*u2*u2 - a23*b12*u1*u2)/a;

            Vector2<T> taus;
            if(!root2real(b,c,taus[0],taus[1])) continue;
            for(T tau:taus)
            {
                if(tau<=0) continue;
                //(tau^2 + b13*tau + 1)*l1^2 = a13
                //positive only
                T l1=std::sqrt(a13/(tau*(tau + b13) + T(1.0)));
                T l3=tau*l1;
                T l2=-(u1*l1 + u3*l3)/u2;
                if(l2<=0) continue;
                Ls[valid]={l1,l2,l3};
                ++valid;
            }
        }
        else
        { // solve for l1


            T w2=T(1.0)/( -u1);
            T w0=u2*w2;
            T w1=u3*w2;

            T a=T(1.0)/((a13 - a12)*w1*w1 - a12*b13*w1 - a12);
            T b=(a13*b12*w1 - a12*b13*w0 - T(2.0)*w0*w1*(a12 - a13))*a;
            T c=((a13 - a12)*w0*w0 + a13*b12*w0 + a13)*a;


            Vector2<T> taus;
            if(!root2real(b,c,taus[0],taus[1])) continue;
            for(T tau:taus)
            {
                if(tau<=0) continue;
                T d=a23/(tau*(b23 + tau) + T(1.0));
                T l2=std::sqrt(d);
                T l3=tau*l2;
                T l1=w0*l2 +w1*l3;
                if(l1<=0) continue;

                Ls[valid]={l1,l2,l3};
                ++valid;
            }
        }
    }


    TOC(lt5);
    TIC(lt6);
    //if constexpr (refinement_iterations>0)
    for(int i=0;i<valid;++i){              gauss_newton_refineL<T,refinement_iterations>(Ls[i],a12,a13,a23,b12,b13,b23);        }
    TOC(lt6);

    TIC(lt8);
    Vector3<T> ry1,ry2,ry3;
    Vector3<T> yd1;
    Vector3<T> yd2;
    Vector3<T> yd1xd2;
    Matrix<T,3,3> X(d12(0),d13(0),d12xd13(0),
                    d12(1),d13(1),d12xd13(1),
                    d12(2),d13(2),d12xd13(2));
    X=X.inverse();

    for(int i=0;i<valid;++i){
        //cout<<"Li="<<Ls(i)<<endl;

        // compute the rotation:
        ry1=y1*Ls(i)(0);
        ry2=y2*Ls(i)(1);
        ry3=y3*Ls(i)(2);

        yd1=ry1-ry2;
        yd2=ry1-ry3;
        yd1xd2=yd1.cross(yd2);

        Matrix<T,3,3> Y(yd1(0),yd2(0),yd1xd2(0),
                        yd1(1),yd2(1),yd1xd2(1),
                        yd1(2),yd2(2),yd1xd2(2));


        Rs[i]=Y*X;
        Ts[i]=(ry1 - Rs[i]*x1 );
        //if(!std::isnan(Ts[solutions][0] + Ts[solutions][1] + Ts[solutions][2]))            solutions++;

        //Ts[i]=(ry1 - Rs[i]*x1 + ry2 - Rs[i]*x2 +ry3 - Rs[i]*x3)/3.0;

    }


    TOC(lt8);
    return valid;







}
}
