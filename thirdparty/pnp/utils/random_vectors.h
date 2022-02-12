#pragma once
/* ********************************* FILE ************************************/
/** \file    random_vectors.h
 *
 * \brief   Uniform sampling on spheres is not trivially obvious, but very useful.
 *
 * \remark
 * - c++11
 * - no dependencies
 * - header only
 *
 *
 *
 *
 *
 * \author   Mikael Persson
 * \date     2010-04-01
 * \note MIT licence
 *
 ******************************************************************************/

#include <utils/random.h>
#include <utils/cvl/matrix.h>
namespace mlib{
/// Uniform distribution on the unit sphere
template<class T,int R> cvl::Vector<T,R> getRandomUnitVector(){
    static_assert(R>1, "not much of a vector otherwize... " );
    cvl::Vector<T,R> n;

    for(int i =0;i<R;++i)
        n[i]=randn<T>(0,1);

    // can happen...
    if(n.abs().sum() <1e-10)
        return getRandomUnitVector<T,R>();

    n.normalize();
    return n;
}
// based on the above && the quaternion properties this gives a uniform distribution of rotations
template<class T=double> cvl::Matrix<T,3,3> getRandomRotation(){
    return getRotationMatrix(getRandomUnitVector<T,4>());
}
}
