#pragma once
/* ********************************* FILE ************************************/
/** \file    rotation_helpers.h
 *
 * \brief    This header contains conversions between various rotation representations and rotation normalization methods
 * \remark
 * - c++11
 * - no dependencies
 * - header only
 * - tested by test_pose.cpp
 *
 * \todo
 * - expand on direct quaternion based rotations
 * - speed comparison between rotation forms
 * - quaternion product to 4x4 matrix, note that the inverse is Q' and that the rotation is Q(0;x)*q'
 * - figure out which euler angle definiton is used
 * - add get rotation from 23d correspondences
 *
 * \author   Mikael Persson
 * \date     2014-04-01
 * \note MIT licence
 *
 ******************************************************************************/
#include <utils/cvl/matrix.h>


namespace cvl{



/**
 * @brief getRotationMatrixX rotation matrix which rotates th radians around the x axis
 * @param th radians
 * @return
 */
template<class T>
Matrix3x3<T> getRotationMatrixX(T th){
    return Matrix3x3<T>(1,0,0,
                        0,cos(th),-sin(th),
                        0,sin(th),cos(th));
}
/**
 * @brief getRotationMatrixY rotation matrix which rotates th radians around the y axis
 * @param th radians
 * @return
 */
template<class T>
Matrix3x3<T> getRotationMatrixY(T th){
    return Matrix3x3<T>(cos(th),0,sin(th),
                        0,1,0,
                        -sin(th),0,cos(th));
}
/**
 * @brief getRotationMatrixZ rotation matrix which rotates th radians around the z axis
 * @param th radians
 * @return
 */
template<class T>
Matrix3x3<T> getRotationMatrixZ(T th){
    return Matrix3x3<T>(cos(th),-sin(th),0,
                        sin(th), cos(th),0,
                        0,       0,1);
}
/**
 * @brief getRotationMatrixXYZ applies in sequence (X(Y(Z
 * @param angle angle x,y,z in radians
 * @return the rotation matrix
 */
template<class T> Matrix3x3<T> getRotationMatrixXYZ(const Vector3<T>& angle){    return getRotationMatrixX(angle[0])*getRotationMatrixY(angle[1])*getRotationMatrixZ(angle[2]);}
template<class T> Matrix3x3<T> getRotationMatrixXYZ(const T& ax, const T& ay, const T& az){    return getRotationMatrixXYZ(Vector3<T>(ax,ay,az));}




// some helper functions
/**
 * @brief normalizeRotationMatrix
 * @param R
 * @return A matrix which is closer to a rotation matrix than the original. Small numerical errors can be compensated this way but large errors should not be corrected this way!
 */
template<class T> Matrix3x3<T> normalizeRotationMatrix(const Matrix3x3<T>& R){




    Vector3D a,b,c;
    a[0]=R(0,0);  b[0]=R(0,1);  //c[0]=R(0,2);
    a[1]=R(1,0);  b[1]=R(1,1);  //c[1]=R(1,2);
    a[2]=R(2,0);  b[2]=R(2,1);  //c[2]=R(2,2);
    a.normalize();
    b=b -  b.dot(a)*a;
    b.normalize();
    c=a.cross(b);




    Matrix3x3D R0(a[0],b[0],c[0],
            a[1],b[1],c[1],
            a[2],b[2],c[2]);

    assert((R-R0).abs().sum()<1e-6);
    return R0;

}




template<class T>
/**
 * @brief quaternionRotate rotates a 3d point using a unit quaternions by implicitly building the rotation matrix first
 * @param q
 * @param x
 * @return
 */
Vector3<T> quaternionRotate(const T* q,const Vector3<T>& x){

    const double t2 =  q[0] * q[1];
    const double t3 =  q[0] * q[2];
    const double t4 =  q[0] * q[3];
    const double t5 = -q[1] * q[1];
    const double t6 =  q[1] * q[2];
    const double t7 =  q[1] * q[3];
    const double t8 = -q[2] * q[2];
    const double t9 =  q[2] * q[3];
    const double t1 = -q[3] * q[3];
    Vector3<T> out(
                (T(2.0) * ((t8 + t1) * x[0] + (t6 - t4) * x[1] + (t3 + t7) * x[2]) + x[0]),
            (T(2.0) * ((t4 + t6) * x[0] + (t5 + t1) * x[1] + (t9 - t2) * x[2]) + x[1]),
            (T(2.0) * ((t7 - t3) * x[0] + (t2 + t9) * x[1] + (t5 + t8) * x[2]) + x[2]));
    return out;
}


template<class T>
/**
 * @brief getEulerAngles returns the euler angles of the rotation matrix according to the definition below
 * @param R
 * @return
 *
 * Definition: ? no freaking idea
 *
 * \todo
 * - figure out the definition
 * - actually just never ever fucking use them, quaternions are universally better!
 */
Vector3<T> getEulerAngles(const Matrix3x3<T>& R){
    // rotation order
    //R(roll)R(pitch)R(yaw) =>
    T pitch=std::asin(-R(2,0));
    T yaw=std::atan(R(2,1)/R(2,2));
    T roll=std::atan(R(1,0)/R(0,0));
    return Vector3<T>(yaw,pitch,roll);
}
template<class T>
__mlib_host_device
/**
 * @brief isRotationMatrix checks if a matrix is a valid rotation matrix
 * @param R
 * @return
 *
 * \todo
 * - verify that the conditions are neccessary and sufficient
 */
bool isRotationMatrix(const Matrix3x3<T>& R){

    if(fabs((R.determinant()-1.0))>1e-7) return false;
    if((R.transpose()*R -Matrix3x3<T>(1,0,0,
                                      0,1,0,
                                      0,0,1)).abs().sum()>1e-7) return false;
    // are the two above sufficient? check later
    return true;
}
template<class T>
__mlib_host_device
/**
 * @brief isRotationQuaternion checks that the vector can be considered a rotation => not nan, not inf, lenth=1
 * @param q
 * @return
 */
bool isRotationQuaternion(const Vector4<T>& q){
    T err=(q.length()-1.0);
    if(err<0)err=-err;
    return err<1e-12;
}

template<class T>
__mlib_host_device
/**
 * @brief isAlmostRotationMatrix checks that a matrix atleast approximatively is a rotation matrix
 * @param R
 * @return
 */
bool isAlmostRotationMatrix(const Matrix3x3<T>& R){
    //if(!(abs((R.determinant()-1.0))<1e-5))
    //cout<<"Not almost Rotation Matrix"<<R<<endl;
    return (std::abs((R.determinant()-1.0))<1e-5);
}





// templated versions:
template<class T>
__mlib_host_device
/**
 * @brief getRotationMatrix x' = R(q)x
 * @param q unit quaternion
 * @return
 */
Matrix3x3<T> getRotationMatrix(const Vector4<T>& q){

    //q.normalize();

    T aa = q[0] * q[0];
    T ab = q[0] * q[1];
    T ac = q[0] * q[2];
    T ad = q[0] * q[3];
    T bb = q[1] * q[1];
    T bc = q[1] * q[2];
    T bd = q[1] * q[3];
    T cc = q[2] * q[2];
    T cd = q[2] * q[3];
    T dd = q[3] * q[3];

    T a=aa + bb - cc - dd;
    T b=T(2.0) * (ad + bc);
    T c=T(2.0) * (bd - ac);
    T d=T(2.0) * (bc - ad);
    T e=aa - bb + cc - dd;
    T f=T(2.0) * (ab + cd);
    T g=  T(2.0) * (ac + bd);
    T h=T(2.0) * (cd - ab);
    T i=aa - bb - cc + dd;
    return Matrix3x3<T>(a, d, g,
                        b,e,h,
                        c,f,i);
}





template<class T>
__mlib_host_device
/**
 * @brief getRotationQuaternion
 * @param R rotation matrix
 * @return
 */
Vector4<T> getRotationQuaternion(const Matrix3x3<T>& R){

    Vector4<T> q;
    //assert(isAlmostRotationMatrix(R));
    T S;

    T tr =R.trace() +1.0;
    if(tr>1e-7){
        S = 0.5 / std::sqrt(tr);
        q[0] = 0.25 / S;
        q[1] = ( R(2,1) - R(1,2) ) * S;
        q[2] = ( R(0,2) - R(2,0) ) * S;
        q[3] = ( R(1,0) - R(0,1) ) * S;

        // does not work for a 45 degree rot around y...
    }else{
        //Matrix3x3D r=R;
        //Matrix3x3D R=r.transpose();
        //throw new std::logic_error("These conversions are probably wrong!");
        //assert(false && "trace zero rotation matrix to quaternion untested");
        //If the trace of the matrix is equal to zero or close then identify which major diagonal element has the greatest value.
        //Depending on this, calculate the following:

        if ( (R(0,0) > R(1,1)) && (R(0,0) > R(2,2)) )  {


            S  = std::sqrt( 1.0 + R(0,0) - R(1,1) - R(2,2) ) * 2.0;
            q[0] = (R(2,1) - R(1,2) ) / S;
            q[1] = 0.25 * S;
            q[2] = (R(1,0) + R(0,1) ) / S;
            q[3] = (R(0,2) + R(2,0) ) / S;


        } else{
            if ( R(1,1) > R(2,2) ) {

                S  = std::sqrt( 1.0 + R(1,1) - R(0,0) - R(2,2) ) * 2.0;
                q[0] = (R(0,2) - R(2,0) ) / S;
                q[1] = (R(1,0) + R(0,1) ) / S;
                q[2] = 0.25 * S;
                q[3] = (R(2,1) + R(1,2) ) / S;
            }
            else {
                S  = std::sqrt( 1.0 + R(2,2) - R(0,0) - R(1,1) ) * 2.0;
                q[0] = (R(1,0) - R(0,1) ) / S;
                q[1] = (R(0,2) + R(2,0) ) / S;
                q[2] = (R(2,1) + R(1,2) ) / S;
                q[3] = 0.25 * S;

            }
        }
    }
    // T err=(q.length()-1);
    // if(err<0)err=-err;






    //assert((err>=1e-4) && "non unit quaternion, ie bad rotation matrix input in R2Q");
   // q.normalize();

    return q;
}



template<class T>
/**
 * @brief QuaternionProduct the standard hamilton/quaternion multiplication.
 * @param z
 * @param w
 * @return
 */
Vector4<T> QuaternionProduct(const Vector4<T>& z, const Vector4<T>& w){
    Vector4<T> zw;
    zw[0] = z[0] * w[0] - z[1] * w[1] - z[2] * w[2] - z[3] * w[3];
    zw[1] = z[0] * w[1] + z[1] * w[0] + z[2] * w[3] - z[3] * w[2];
    zw[2] = z[0] * w[2] - z[1] * w[3] + z[2] * w[0] + z[3] * w[1];
    zw[3] = z[0] * w[3] + z[1] * w[2] - z[2] * w[1] + z[3] * w[0];
    return zw;
}


template<class T>
/**
 * @brief conjugateQuaternion
 * @param q
 * @return the conjugated quaternion
 */
Vector4<T> conjugateQuaternion(const Vector4<T>& q){
    Vector4<T> qi;
    qi[0]=q[0];
    qi[1]=-q[1];
    qi[2]=-q[2];
    qi[3]=-q[3];
    // must be equivalent to q[0]=-q[0] right? only for unit quaternions
    return qi;
}
template<class T>
/**
 * @brief invertQuaternion
 * @param q
 * @return the inverse quaternion
 */
Vector4<T> invertQuaternion(const Vector4<T>& q){
    Vector4<T> qi;
    qi[0]=q[0];
    qi[1]=-q[1];
    qi[2]=-q[2];
    qi[3]=-q[3];

    return qi/qi.squaredLength(); // should it really be squared? wierd
}

template<class T>
/**
 * @brief QuaternionRotate
 * @param z a unit quaternion
 * @param w
 * @return
 */
Vector3<T> QuaternionRotate(const Vector4<T>& z, const Vector3<T>& w){


    Vector4<T> r(T(0),w[0],w[1],w[2]);
    r=QuaternionProduct(z,r);
    r=QuaternionProduct(r,conjugateQuaternion(z));



    return Vector3<T>(r[1],r[2],r[3]);
}





}//end namespace cvl
