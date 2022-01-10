/* 
 * imu_tk - Inertial Measurement Unit Toolkit
 * 
 *  Copyright (c) 2014, Alberto Pretto <pretto@diag.uniroma1.it>
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 * 
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 * 
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <vector>
#include "imu_tk/base.h"

namespace imu_tk
{

/**
  * @brief Classify between static and motion intervals checking if for each sample 
  *        of the input signal \f$s\f$ (samples) the local variance magnitude 
  *        is lower or greater then a threshold.
  * 
  * @param samples Input 3D signal (e.g, the acceleremeter readings)
  * @param threshold Threshold used in the classification
  * @param[out] intervals  Ouput detected static intervals
  * @param win_size Size of the sliding window (i.e., number of samples) 
  *                 used to compute the local variance magnitude. It should be equal or
  *                 greater than 11
  * 
  * 
  * The variance magnitude is a scalar computed in a temporal sliding window of size 
  * \f$w_s\f$ (i.e., win_size) as:
  * \f[ \varsigma(t) = 
  *     \sqrt{[var_{w_s}(s^t_x)]^2 + [var_{w_s}(s^t_y)]^2 + [var_{w_s}(s^t_z)]^2} \f] 
  * 
  * Where \f$var_{w_s}(s^t)\f$ is an operator that compute the variance of
  * a general 1D signal in a interval of length \f$w_s\f$ samples
  * centered in \f$t\f$.
  */
template <typename _T> 
  void staticIntervalsDetector ( const std::vector< TriadData_<_T> > &samples,
                                 _T threshold, std::vector< DataInterval > &intervals,
                                 int win_size = 101 );
}
