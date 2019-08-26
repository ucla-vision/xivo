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

#include <Eigen/Core>
#include <ceres/rotation.h>

#include "imu_tk/base.h"

#include <iostream>

namespace imu_tk
{
/** @brief Normalize a input quaternion to an unit vector 
 * 
 * @param[in,out] quat The Eigen 4D vector representing the quaternion to be normlized
 */
template <typename _T> inline void normalizeQuaternion( Eigen::Matrix< _T, 4, 1> &quat );

/** @brief Normalize a input quaternion to an unit vector 
 * 
 * @param[in,out] quat The 4D array representing the quaternion to be normlized
 */
template <typename _T> inline void normalizeQuaternion( _T quat[4] );

/** @brief Perform a RK4 Runge-Kutta integration step
 * 
 * @param quat The input Eigen 4D vector representing the initial rotation
 * @param omega0 Initial rotational velocity at time t0
 * @param omega1 Final rotational velocity at time t1
 * @param dt Time step (t1 - t0).
 * @param[out] quat_res Resulting final rotation
 */
template <typename _T> inline void quatIntegrationStepRK4( const Eigen::Matrix< _T, 4, 1> &quat, 
                                                           const Eigen::Matrix< _T, 3, 1> &omega0, 
                                                           const Eigen::Matrix< _T, 3, 1> &omega1, 
                                                           const _T &dt, Eigen::Matrix< _T, 4, 1> &quat_res );

/** @brief Perform a RK4 Runge-Kutta integration step
 * 
 * @param quat The input 4D array representing the initial rotation
 * @param omega0 Initial rotational velocity at time t0
 * @param omega1 Final rotational velocity at time t1
 * @param dt Time step (t1 - t0).
 * @param[out] quat_res Resulting final rotation
 */
template <typename _T> inline void quatIntegrationStepRK4( const _T quat[4], 
                                                           const _T omega0[3], 
                                                           const _T omega1[3], 
                                                           const _T &dt, _T quat_res[4] );

/** @brief Integrate a sequence of rotational velocities using the RK4 
 *         Runge-Kutta discrete integration method. The initial rotation is assumed to be the
 *         identity quaternion.
 * 
 * @param gyro_samples Input gyroscope signal (rotational velocity samples vector)
 * @param[out] quat_res Resulting final rotation quaternion
 * @param dt Fixed time step (t1 - t0) between samples. If is -1, the sample timestamps are used instead.
 * @param interval Data interval where to compute the integration. If this interval is not valid,
 *                 i.e., one of the two indices is -1, the integration is computed for the whole data
 *                 sequence.
 */
template <typename _T> void integrateGyroInterval( const std::vector< TriadData_<_T> > &gyro_samples, 
                                                   Eigen::Matrix< _T, 4, 1> &quat_res, _T data_dt = _T(-1),
                                                   const DataInterval &interval = DataInterval() );

/** @brief Integrate a sequence of rotational velocities using the RK4 
 *         Runge-Kutta discrete integration method. The initial rotation is assumed to be the
 *         identity rotation matrix.
 * 
 * @param gyro_samples Input gyroscope signal (rotational velocity samples vector)
 * @param[out] quat_res Resulting final rotation matrix
 * @param dt Fixed time step (t1 - t0) between samples. If is -1, the sample timestamps are used instead.
 * @param interval Data interval where to compute the integration. If this interval is not valid,
 *                 i.e., one of the two indices is -1, the integration is computed for the whole data
 *                 sequence.
 */
template <typename _T> void integrateGyroInterval( const std::vector< TriadData_<_T> > &gyro_samples, 
                                                   Eigen::Matrix< _T, 3, 3> &rot_res, _T data_dt = _T(-1),
                                                   const DataInterval &interval = DataInterval() );

}

/* Implementation */

template <typename _T> inline void imu_tk::normalizeQuaternion ( Eigen::Matrix< _T, 4 , 1  >& quat )
{
  _T quat_norm = quat.norm();
  quat /= quat_norm;
}

template <typename _T> inline void imu_tk::normalizeQuaternion ( _T quat[4] )
{
  Eigen::Matrix< _T, 4 , 1  > tmp_q = Eigen::Map< Eigen::Matrix< _T, 4 , 1  > >(quat);
  imu_tk::normalizeQuaternion ( tmp_q );
}

template <typename _T> 
  static inline void computeOmegaSkew( const Eigen::Matrix< _T, 3, 1> &omega, 
                                       Eigen::Matrix< _T, 4, 4> &skew )
{
  skew <<   _T(0),     -omega(0),  -omega(1),  -omega(2),
            omega(0),   _T(0),      omega(2),  -omega(1),
            omega(1),  -omega(2),   _T(0),      omega(0),
            omega(2),   omega(1),  -omega(0),   _T(0);
}

template <typename _T> 
  inline void imu_tk::quatIntegrationStepRK4( const Eigen::Matrix< _T, 4, 1> &quat, 
                                              const Eigen::Matrix< _T, 3, 1> &omega0, 
                                              const Eigen::Matrix< _T, 3, 1> &omega1, 
                                              const _T &dt, Eigen::Matrix< _T, 4, 1> &quat_res )
{
  Eigen::Matrix< _T, 3, 1> omega01 = _T(0.5)*( omega0 + omega1 );
  Eigen::Matrix< _T, 4, 1> k1, k2, k3, k4, tmp_q;
  Eigen::Matrix< _T, 4, 4> omega_skew;
  
  // First Runge-Kutta coefficient
  computeOmegaSkew( omega0, omega_skew );
  k1 = _T(0.5)*omega_skew*quat;
  // Second Runge-Kutta coefficient
  tmp_q = quat + _T(0.5)*dt*k1;
  computeOmegaSkew( omega01, omega_skew );
  k2 = _T(0.5)*omega_skew*tmp_q;
  // Third Runge-Kutta coefficient (same omega skew as second coeff.)
  tmp_q = quat + _T(0.5)*dt*k2;
  k3 = _T(0.5)*omega_skew*tmp_q;
  // Forth Runge-Kutta coefficient
  tmp_q = quat + dt*k3;
  computeOmegaSkew( omega1, omega_skew );
  k4 = _T(0.5)*omega_skew*tmp_q;
  _T mult1 = _T(1.0)/_T(6.0), mult2 = _T(1.0)/_T(3.0);
  quat_res = quat + dt*(mult1*k1 + mult2*k2 + mult2*k3 + mult1*k4);
  normalizeQuaternion(quat_res);
}

template <typename _T> 
  inline void imu_tk::quatIntegrationStepRK4( const _T quat[4], const _T omega0[3], const _T omega1[3], 
                                              const _T &dt, _T quat_res[4] )
{
  const Eigen::Matrix< _T, 4, 1> m_quat = Eigen::Map< const Eigen::Matrix< _T, 4, 1> >(quat);
  const Eigen::Matrix< _T, 3, 1> m_omega0 = Eigen::Map< const Eigen::Matrix< _T, 3, 1> >(omega0),
                                 m_omega1 = Eigen::Map< const Eigen::Matrix< _T, 3, 1> >(omega1); 
  Eigen::Matrix< _T, 4, 1> m_quat_res;
  
  quatIntegrationStepRK4( m_quat, m_omega0, m_omega1, dt, m_quat_res );
  
  quat_res[0] = m_quat_res(0);
  quat_res[1] = m_quat_res(1);
  quat_res[2] = m_quat_res(2);
  quat_res[3] = m_quat_res(3);
}

template <typename _T> void imu_tk::integrateGyroInterval( const std::vector< TriadData_<_T> > &gyro_samples, 
                                                           Eigen::Matrix< _T, 4, 1> &quat_res,
                                                           _T data_dt, const DataInterval &interval )
{
  DataInterval rev_interval =  checkInterval( gyro_samples, interval );

  quat_res = Eigen::Matrix< _T, 4, 1>(_T(1.0), _T(0), _T(0), _T(0)); // Identity quaternion
  
  for( int i = rev_interval.start_idx; i < rev_interval.end_idx; i++)
  {
    _T dt = ( data_dt > _T(0))?data_dt:gyro_samples[i+1].timestamp() - gyro_samples[i].timestamp();
    
    quatIntegrationStepRK4( quat_res,
                            gyro_samples[i].data(), 
                            gyro_samples[i + 1].data(), 
                            dt,
                            quat_res );
  }
}

template <typename _T> void imu_tk::integrateGyroInterval( const std::vector< TriadData_<_T> >& gyro_samples, 
                                                           Eigen::Matrix< _T, 3 , 3  >& rot_res, 
                                                           _T data_dt, const DataInterval& interval )
{
  Eigen::Matrix< _T, 4, 1> quat_res;
  integrateGyroInterval( gyro_samples, quat_res, data_dt, interval );
  ceres::MatrixAdapter<_T, 1, 3> rot_mat = ceres::ColumnMajorAdapter3x3(rot_res.data());
  ceres::QuaternionToRotation( quat_res.data(), rot_mat );
}