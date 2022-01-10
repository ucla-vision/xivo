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

#include <string>
#include <stdint.h>
#include <boost/shared_ptr.hpp>

#include "imu_tk/base.h"


namespace imu_tk
{
  
class Plot
{
public:
  Plot();
  ~Plot(){};
  
  template <typename _T> 
    void plotSamples( const std::vector< TriadData_<_T> > &samples,
                      DataInterval range = DataInterval() );
  template <typename _T> 
    void plotIntervals( const std::vector< TriadData_<_T> > &samples,
                        const std::vector< DataInterval > &intervals,
                        DataInterval range = DataInterval() );
private:

  /* Pimpl idiom */
  class PlotImpl; 
  boost::shared_ptr< PlotImpl > plot_impl_ptr_;
};

void waitForKey();


class Vis3D
{
public:
  
  Vis3D( const std::string win_name = "imu_tk" );
  ~Vis3D(){};
  
  void registerFrame( std::string name, uint8_t r = 255, uint8_t g = 255, uint8_t b = 255 );
  void unregisterFrame( std::string name );    
  template <typename _T> 
    void setFramePos( std::string name, const _T quat[4], const _T t[3] );

  void registerLine( std::string name, uint8_t r = 255, uint8_t g = 255, uint8_t b = 255 );
  void unregisterLine( std::string name );    
  template <typename _T> 
    void setLinePos( std::string name, const _T p0[3], const _T p1[3] );
    
  void updateAndWait( int delay_ms = 0 );
    
private:
  
  /* Pimpl idiom */
  class VisualizerImpl;
  boost::shared_ptr< VisualizerImpl > vis_impl_ptr_;
};


}