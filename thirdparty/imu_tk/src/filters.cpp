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

#include "imu_tk/filters.h"

using namespace Eigen;

template <typename _T> 
  void imu_tk::staticIntervalsDetector ( const std::vector< imu_tk::TriadData_<_T> >& samples, 
                                         _T threshold, std::vector< imu_tk::DataInterval >& intervals, 
                                         int win_size )
{
  if ( win_size < 11 ) win_size = 11;
  if( !(win_size % 2) ) win_size++;
  
  int h = win_size / 2;
  
  if( win_size >=  samples.size() )
    return;
 
  intervals.clear();
  
  bool look_for_start = true;
  imu_tk::DataInterval current_interval;
  
  for( int i = h; i < samples.size() - h; i++ )
  {
    Matrix< _T, 3, 1> variance = dataVariance( samples, DataInterval( i - h, i + h) );
    _T norm = variance.norm();
    
    if( look_for_start )
    {
      if (norm < threshold)
      {
        current_interval.start_idx = i;
        look_for_start = false;
      }
    }
    else
    {
      if (norm >= threshold)
      {
        current_interval.end_idx = i - 1;
        look_for_start = true;
        intervals.push_back(current_interval);
      }
    }
  }
  
  // If the last interval has not been included in the intervals vector
  if( !look_for_start )
  {
    current_interval.end_idx = samples.size() - h - 1;
    //current_interval.end_ts = samples[current_interval.end_idx].timestamp();
    intervals.push_back(current_interval);
  }
}

template void imu_tk::staticIntervalsDetector<double> ( const std::vector< TriadData_<double> > &samples,
                                                        double threshold, std::vector< DataInterval > &intervals,
                                                        int win_size = 101 );
template void imu_tk::staticIntervalsDetector<float> ( const std::vector< TriadData_<float> > &samples,
                                                       float threshold, std::vector< DataInterval > &intervals,
                                                       int win_size = 101 );
