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
enum TimestampUnit
{
  TIMESTAMP_UNIT_SEC = 1,
  TIMESTAMP_UNIT_MSEC = 1000,
  TIMESTAMP_UNIT_USEC = 1000000,
  TIMESTAMP_UNIT_NSEC = 1000000000
};

enum DatasetType
{
  DATASET_SPACE_SEPARATED,
  DATASET_COMMA_SEPARATED
};

template <typename _T> 
  void importAsciiData( const char *filename,
                        std::vector< TriadData_<_T> > &samples, 
                        TimestampUnit unit = TIMESTAMP_UNIT_USEC,
                        DatasetType type = DATASET_SPACE_SEPARATED );
template <typename _T> 
  void importAsciiData( const char *filename,
                        std::vector< TriadData_<_T> > &samples0,
                        std::vector< TriadData_<_T> > &samples1, 
                        TimestampUnit unit = TIMESTAMP_UNIT_USEC,
                        DatasetType type = DATASET_SPACE_SEPARATED );
template <typename _T> 
  void importAsciiData( const char *filename,
                        std::vector< TriadData_<_T> > &samples0,
                        std::vector< TriadData_<_T> > &samples1,
                        std::vector< TriadData_<_T> > &samples2, 
                        TimestampUnit unit = TIMESTAMP_UNIT_USEC,
                        DatasetType type = DATASET_SPACE_SEPARATED ); 
}