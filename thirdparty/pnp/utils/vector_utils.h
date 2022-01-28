#pragma once
/* ********************************* FILE ************************************/
/** \file   vector.h
 *
 * \brief    This header contains common convenience functions for std::vector
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
 * \date     2016-04-01
 * \note MIT licence
 *
 ******************************************************************************/




#ifndef VECTOR_HELPERS_H
#define VECTOR_HELPERS_H
/** vector.h contains common functions for the std::vector
 * and common functions shared with std::set and std::map
 * */










#include <vector>
#include <set>
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <map>


///////////////////////////// SELF CONTAINED //////////////////////////////







namespace mlib {




/**
 * @brief equal are the two vectors the same size and their elements equal
 * @param a
 * @param b
 * @return
 */
template<class T> bool equal(const std::vector<T>& a, const std::vector<T>& b){
    if(a.size()!=b.size()) return false;
    for(uint i=0;i<a.size();++i){
        if(a[i]!=b[i]) return false;
    }
    return true;
}


// vector helpers

/**
 * @brief mean
 * @param v
 * @return vector mean of v
 */
template<typename T>
T mean(const std::vector<T>& v)
{
    assert(v.size() > 0 && "mean of zero element vector requested");
    T m = v.at(0);
    for (uint i = 1; i < v.size(); ++i) {
        m += v[i];
    }
    return m /=  v.size();
}

/**
 * @brief variance
 * @param v
 * @param meanv
 * @return E((X- E(X))^2)
 *
 * NOTE: this is the variance, not the sample variance
 *
 * the unbiased estimate of the variance is the sample_variance=(n/(n-1)variance(x)
 *
 * so if used when sample variance is intended this function underestimates the variance
 *
 * In other words the mean must be known and exact!
 *
*/
template<typename T>
T variance(const std::vector<T>& v, const T& meanv)
{


    assert(v.size() > 1 && "variance of zero element vector requested");
    if(v.size()==0) return 	std::numeric_limits<T>::infinity();

    T var = T(0);
    for (const T& e : v) {
        var += (e - meanv) * (e - meanv);
    }

    return var /  double(v.size());

}

template<typename T>
/**
 * @brief variance
 * @param v
 * @return the sample variance of (v)
 */
T variance(const std::vector<T>& v)   {
    return ((double)v.size()/((double)v.size()-1))*variance(v,mean(v));
}
template<class T> bool hasNan(const std::vector<T>& xs)
{
    for (const T& x : xs)
        if (std::isnan(x)) return true;
    return false;
}



template<typename T>
/**
 * @brief median returns the vector median, underestimates if the size is even
 * @param v
 * @return
 *
 * \todo
 * - replace with the fast median
 */
T median(std::vector<T> v)
{
    assert((v.size() > 0) && "median of zero element vector requested");
    if(v.size()==0) return std::numeric_limits<T>::quiet_NaN();
    if (v.size() == 1)        return v[0];
    sort(v.begin(), v.end());

    return v.at((unsigned int)std::round(v.size() / 2));
}
///@return the smallest or throws, @param v
template<typename T> T min(const std::vector<T>& v)
{
    assert((v.size() != 0) && "min of empty vector undefined");
    T m = v.at(0);
    for (uint i = 1; i < v.size(); ++i) {
        if (m > v[i])
            m = v[i];
    }
    return m;
}
///@return the largest or throws in vector @param v
template<typename T> T max(const std::vector<T>& v)
{
    assert((v.size() != 0) && "max of empty vector undefined");
    T m = v.at(0);
    for (uint i = 1; i < v.size(); ++i) {
        if (m < v[i])
            m = v[i];
    }
    return m;
}
/// returns the largest and smallest or throws
template<typename T> void minmax(const std::vector<T>& v /** @param v vector*/,
                                 T& minv /** @param minv*/,
                                 T& maxv /** @param maxv */)
{
    assert((v.size() != 0) && "minmax of empty vector undefined");
    minv = v.at(0);
    maxv = v.at(0);
    for (uint i = 1; i < v.size(); ++i) {
        if (maxv < v[i])
            maxv = v[i];
        if (minv > v[i])
            minv = v[i];
    }
}



///@return the sum of the vector @param v
template<typename T> T sum(const std::vector<T>& v)
{
    assert((v.size() != 0) && "sum of empty vector undefined");
    T m = v.at(0);
    for (uint i = 1; i < v.size(); ++i) {
        m += v[i];
    }
    return m;
}
/// applies abs to each element of the vector @param v
template<typename T> void abs(std::vector<T>& v)
{
    for (int i = 0; i < v.size(); ++i)
        v[i] = std::abs(v[i]);
}
/// squares each element, not norm2 of @param v
template<typename T> void square(std::vector<T>& v)
{
    for (int i = 0; i < v.size(); ++i)
        v[i] = v[i] * v[i];
}
/**
 * @brief unique_filter removes duplicate elements
 * @param vs
 * @return
 */
template<class T>
std::vector<T> unique_filter(const std::vector<T>& vs)
{
    std::set<T> us; std::vector<T> rs; rs.reserve(vs.size());
    for (const auto& v : vs)
        us.insert(v);
    for (const auto& u : us)
        rs.push_back(u);
    return rs;
}
/// reverses the element order in the vector @param v
template<typename T> void reverse(std::vector<T>& v) {
    std::vector<T> r; r.reserve(v.size());
    for (int i = v.size() - 1; i > -1; --i) {
        r.push_back(v[i]);
    }
    v = r;
}


template<typename T>
/**
 * @brief keep_filter keeps the elements for which keep evaluates to true
 * @param keep
 * @param v
 */
void keep_filter(const std::vector<bool>& keep,
                                      std::vector<T>& v)
{
    assert(v.size() == keep.size());
    std::vector<T> vf; vf.reserve(v.size());
    for (uint i = 0; i < v.size(); ++i) {
        if (keep[i])
            vf.push_back(v[i]);
    }
    v = vf;
}
/**
 * @brief filterKeepIfAandB keeps if both a and b conditions are valid
 * @param AandB_A
 * @param AandB_B
 * @param v
 */
template<typename T>
void filterKeepIfAandB(const std::vector<bool>& AandB_A,const std::vector<bool>& AandB_B, std::vector<T>& v)
{
    assert(v.size() == AandB_A.size());
    assert(v.size() == AandB_B.size());
    std::vector<T> vf; vf.reserve(v.size());
    for (uint i = 0; i < v.size(); ++i) {
        if (AandB_A[i] && AandB_B[i] )
            vf.push_back(v[i]);
    }
    v = vf;
}

/**
 * @brief remove_filter removes specified elments
 * @param v
 * @param remove
 */
template<typename T> void remove_filter(std::vector<T>& v, std::vector<T>& remove)
{
    std::set<T> u;
    std::vector<T> r; r.reserve(v.size());

    for (uint i = 0; i < remove.size(); ++i)
        u.insert(remove[i]);


    for (uint i = 0; i < v.size(); ++i) {
        if (u.count(v[i]) == 0)
            r.push_back(v[i]);
    }
    v = r;
}
template<typename T>
/**
 * @brief operator + element wise addition
 * @param lhs
 * @param rhs
 * @return
 */
std::vector<T> operator+ (const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    assert(lhs.size() == rhs.size());
    std::vector<T> r; r.reserve(lhs.size());
    for (uint i = 0; i < lhs.size() && i<rhs.size(); ++i)
        r.push_back(lhs[i] + rhs[i]);
    return r;
}
///@return the set as a vector
template<typename T> std::vector<T> fromSet(const std::set<T>& st /** @param st the set*/)
{
    std::vector<T> tmp; tmp.reserve(st.size());
    for (auto it = st.begin(); it != st.end(); ++it)
        tmp.push_back(*it);
    return tmp;
}


template<class T>
/**
 * @brief isSorted
 * @param ys
 * @return true if the vector is sorted, asc order
 */
bool isSorted(const std::vector<T>& ys){

    for(int i=1;i<ys.size();++i) if(ys[i-1]>ys[i]) return false;
    return true;
}

template<class T>
/**
 * @brief AandB elementwise and, if a vector is too short the rest is assumed false
 * @param a
 * @param b
 * @return
 */
std::vector<T> AandB(const std::vector<T>& a,
                     const std::vector<T>& b){
    std::vector<T> out; out.reserve(a.size());
    for(uint i=0;i<a.size() && i<b.size();++i)
        out.push_back(a[i] && b[i]);
    return out;
}

template<class T>
/**
 * @brief count the number of elements which evaluate to true in vector(v)
 * @param v
 * @return
 */
int count(const std::vector<T>& v){int c=0;for(auto b:v) if(b)++c;return c;}









}// end namespace mlib
#endif // VECTOR_HELPERS_H
