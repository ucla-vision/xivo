#pragma once

/* ********************************* FILE ************************************/
/** \file    random.h
 *
 * \brief    This header contains contains convenience functions for repetable random values
 *
 * \remark
 * - c++11
 * - no dependencies
 * - self contained(just .h,.cpp)
 * - C++11 accurate generators are consistently faster and better than rand ever was!
 * - Repeatability is guaranteed if the seed is set and only these functions are used.
 *
 *
 *
 * \todo
 *
 * - convert to pure header for ease of inclusion and to ensure current not original flags are used!
 *
 *  Cmake Options: sets the flags
 * -DRANDOM_SEED_VALUE=0
 * -DRANDOM_SEED_FROM_TIME ON
 *  * option:
 *
 *
 *  * RANDOM DEFAULTS TO
 * RANDOM_SEED_VALUE 0
 * RANDOM_SEED_FROM_TIME OFF
 *
 *  Note, there is no way to have synch values for a multithreaded system.
 *
 *
 * \author   Mikael Persson
 * \date     2007-04-01
 * \note MIT licence
 *
 ******************************************************************************/

#ifndef RANDOM_SEED_VALUE
#define RANDOM_SEED_VALUE 0
#endif




//////////////////// SELF CONTAINED ////////////////////////////




#include <random>
#include <vector>
#include <set>
#include <mutex>
#include <algorithm>

namespace mlib{




namespace random{

/// the random generator mutex,
static std::mutex gen_mtx;
/// shall leak
static std::default_random_engine generator;
static bool seeded=false;


template<int V>  void init_common_generator(){
    if(seeded) return; // dont lock unless you need to
    std::unique_lock<std::mutex> ul(gen_mtx); // lock
    if(seeded) return; // what if someone fixed it in the mean time?
    seeded=true;
    generator=std::default_random_engine();

#ifdef RANDOM_SEED_FROM_TIME
    unsigned long int seed=static_cast<unsigned long>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
    generator.seed(seed);
#else
    generator.seed(RANDOM_SEED_VALUE);
#endif

    /// I have avoided using local static since that does not work for old compilers or sometimes new ones...
}


} // end namespace random

/**
 * @brief randu integer random value drawn from uniform distribution
 * @param low  - included
 * @param high - included
 * @return random value according to compile time random value strategy
 */
template<class T> T randu(T low=0, T high=1){
    static_assert(std::is_floating_point<T>::value,          "template argument not a floating point type");
    random::init_common_generator<0>();
    std::uniform_real_distribution<T> rn(low,high);
    return rn(random::generator);
}
/**
 * @brief randui integer random value drawn from uniform distribution
 * @param low  - included
 * @param high - included
 * @return random value according to compile time random value strategy
 */
template<class T> T randui(T low=0, T high=1){
    random::init_common_generator<0>();
    std::uniform_int_distribution<T> rn(low,high);
    return rn(random::generator);
}
/**
 * @brief randn random value drawn from normal distribution
 * @param m
 * @param sigma
 * @return random value drawn from normal distribution
 */
template<class T> T randn(T mean=0, T sigma=1){
    static_assert(std::is_floating_point<T>::value,          "template argument not a floating point type");
    random::init_common_generator<0>();
    std::normal_distribution<T> rn(mean, sigma);
    return rn(random::generator);
}

} // end namespace mlib
