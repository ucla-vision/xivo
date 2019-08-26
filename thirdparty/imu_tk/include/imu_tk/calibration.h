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
#include <iostream>
#include <fstream>

#include "imu_tk/base.h"

namespace imu_tk
{
/** @brief This object contains the calibration parameters (misalignment, scale factors, ...)
 *         of a generic orthogonal sensor triad (accelerometers, gyroscopes, etc.)
 * 
 * Triad model:
 *         
 * -Misalignment matrix:
 * 
 * general case:
 * 
 *     [    1     -mis_yz   mis_zy  ]
 * T = [  mis_xz     1     -mis_zx  ]
 *     [ -mis_xy   mis_yx     1     ]
 * 
 * "body" frame spacial case:
 * 
 *     [  1     -mis_yz   mis_zy  ]
 * T = [  0        1     -mis_zx  ]
 *     [  0        0        1     ]
 * 
 * Scale matrix:
 * 
 *     [  s_x      0        0  ]
 * K = [   0      s_y       0  ]
 *     [   0       0       s_z ]
 * 
 * Bias vector:
 * 
 *     [ b_x ]
 * B = [ b_y ]
 *     [ b_z ]
 * 
 * Given a raw sensor reading X (e.g., the acceleration ), the calibrated "unbiased" reading X' is obtained
 * 
 * X' = T*K*(X - B)
 * 
 * with B the bias (variable) + offset (constant, possibbly 0), or, equivalently:
 * 
 * X' = T*K*X - B'
 * 
 * with B' = T*K*B
 * 
 * Without knowing the value of the bias (and with offset == 0), the calibrated reading X'' is simply:
 * 
 * X'' = T*K*X
*/
template < typename _T > class CalibratedTriad_
{
public:
  /** @brief Basic "default" constructor: without any parameter, it initilizes the calibration parameter with 
   *         default values (zero scaling factors and biases, identity misalignment matrix)
   */
  CalibratedTriad_( const _T &mis_yz = _T(0), const _T &mis_zy = _T(0), const _T &mis_zx = _T(0), 
                    const _T &mis_xz = _T(0), const _T &mis_xy = _T(0), const _T &mis_yx = _T(0), 
                    const _T &s_x = _T(1),    const _T &s_y = _T(1),    const _T &s_z = _T(1), 
                    const _T &b_x = _T(0),    const _T &b_y = _T(0),    const _T &b_z  = _T(0) );
 
  ~CalibratedTriad_(){};
               
  inline _T misYZ() const { return -mis_mat_(0,1); };
  inline _T misZY() const { return mis_mat_(0,2); };
  inline _T misZX() const { return -mis_mat_(1,2); };
  inline _T misXZ() const { return mis_mat_(1,0); };
  inline _T misXY() const { return -mis_mat_(2,0); };
  inline _T misYX() const { return mis_mat_(2,1); };

  inline _T scaleX() const { return scale_mat_(0,0); };
  inline _T scaleY() const { return scale_mat_(1,1); };
  inline _T scaleZ() const { return scale_mat_(2,2); };
      
  inline _T biasX() const { return bias_vec_(0); };
  inline _T biasY() const { return bias_vec_(1); };
  inline _T biasZ() const { return bias_vec_(2); };
  
  inline const Eigen::Matrix< _T, 3 , 3>& getMisalignmentMatrix() const { return mis_mat_; };
  inline const Eigen::Matrix< _T, 3 , 3>& getScaleMatrix() const { return scale_mat_; };
  inline const Eigen::Matrix< _T, 3 , 1>& getBiasVector() const { return bias_vec_; };
    
  inline void setScale( const Eigen::Matrix< _T, 3 , 1> &s_vec ) 
  { 
    scale_mat_(0,0) = s_vec(0); scale_mat_(1,1) = s_vec(1);  scale_mat_(2,2) = s_vec(2); 
    update();
  };
  
  inline void setBias( const Eigen::Matrix< _T, 3 , 1> &b_vec ) 
  { 
    bias_vec_ = b_vec;
    update();
  };
  
  /** @brief Load the calibration parameters from a simple text file.
   * 
   * The file should containts a sequence of two, space separated 3X3 matrixes 
   * (the misalignment and the scale matrix) followed by a 3x1 biases vector (see the load()
   * function)
   */
  bool load( std::string filename );
  
  /** @brief Save the calibration parameters in a simple text file.
   * 
   * The file will containts a sequence of two, space separated 3X3 matrixes 
   * (the misalignment and the scale matrix) followed by a 3x1 biases vector 
   */
  bool save( std::string filename ) const;

  /** @brief Normalize a raw data X by correcting the misalignment and the scale,
   *         i.e., by applying the equation  X'' = T*K*X
   */
  inline Eigen::Matrix< _T, 3 , 1> normalize( const Eigen::Matrix< _T, 3 , 1> &raw_data ) const
  {
    return ms_mat_*raw_data;
  };
  
  /** @brief Normalize a raw data X by correcting the misalignment and the scale,
   *         i.e., by applying the equation  X'' = T*K*X
   */
  inline TriadData_<_T> normalize( const TriadData_<_T> &raw_data ) const
  {
    return TriadData_<_T>( raw_data.timestamp(), normalize( raw_data.data()) );
  };
  
  /** @brief Normalize a raw data X by removing the biases and 
   *         correcting the misalignment and the scale, 
   *         i.e., by applying the equation  X' = T*K*(X - B)
   */
  inline Eigen::Matrix< _T, 3 , 1> unbiasNormalize( const Eigen::Matrix< _T, 3 , 1> &raw_data ) const
  {
    return ms_mat_*(raw_data - bias_vec_); 
  };
  
  /** @brief Normalize a raw data X by removing the biases and 
   *         correcting the misalignment and the scale, 
   *         i.e., by applying the equation  X' = T*K*(X - B)
   */
  inline TriadData_<_T> unbiasNormalize( const TriadData_<_T> &raw_data ) const
  {
    return TriadData_<_T>( raw_data.timestamp(), unbiasNormalize( raw_data.data()) );
  };
  
  /** @brief Remove the biases from a raw data */
  inline Eigen::Matrix< _T, 3 , 1> unbias( const Eigen::Matrix< _T, 3 , 1> &raw_data ) const
  {
    return raw_data - bias_vec_; 
  };
  
  /** @brief Remove the biases from a raw data */
  inline TriadData_<_T> unbias( const TriadData_<_T> &raw_data ) const
  {
    return TriadData_<_T>( raw_data.timestamp(), unbias( raw_data.data()) );
  };
  
private:

  /** @brief Update internal data (e.g., compute Misalignment * scale matrix) 
   *         after a parameter is changed */
  void update();
  
  /** @brief Misalignment matrix */
  Eigen::Matrix< _T, 3 , 3> mis_mat_;
  /** @brief Scale matrix */
  Eigen::Matrix< _T, 3 , 3> scale_mat_;
  /** @brief Bias vector */
  Eigen::Matrix< _T, 3 , 1> bias_vec_;
  /** @brief Misalignment * scale matrix */
  Eigen::Matrix< _T, 3 , 3> ms_mat_;
};

typedef CalibratedTriad_<double> CalibratedTriad;

/** @brief Generates a sequence of characters with a properly formatted 
 *         representation of a CalibratedTriad_  instance (calib_triad), 
 *         and inserts them into the output stream os. */
template <typename _T> std::ostream& operator<<(std::ostream& os, 
                                                const imu_tk::CalibratedTriad_<_T>& calib_triad);

/** @brief This object enables to calibrate an accelerometers triad and eventually
 *         a related gyroscopes triad (i.e., to estimate theirs misalignment matrix, 
 *         scale factors and biases) using the multi-position calibration method.
 * 
 * For more details, please see:
 * 
 * D. Tedaldi, A. Pretto and E. Menegatti 
 * "A Robust and Easy to Implement Method for IMU Calibration without External Equipments"
 * In: Proceedings of the IEEE International Conference on Robotics and Automation (ICRA 2014), 
 * May 31 - June 7, 2014 Hong Kong, China, Page(s): 3042 - 3049 
 */
template <typename _T> class MultiPosCalibration_
{
public:
  
  /** @brief Default constructor: initilizes all the internal members with default values */
  MultiPosCalibration_();
  ~MultiPosCalibration_(){};
  
  /** @brief Provides the magnitude of the gravitational filed 
   *         used in the calibration (i.e., the gravity measured in
   *         the place where the calibration dataset has been acquired) */
  _T gravityMagnitede() const { return g_mag_; };

  /** @brief Provides the duration in seconds of the initial static interval */
  _T initStaticIntervalDuration() const { return init_interval_duration_; };
  
  /** @brief Provides the number of data samples to be extracted from each detected static intervals */
  int intarvalsNumSamples() const { return interval_n_samples_; };
  
  /** @brief Provides the accelerometers initial guess calibration parameters */
  const CalibratedTriad_<_T>& initAccCalibration(){ return init_acc_calib_; };
  
  /** @brief Provides the gyroscopes initial guess calibration parameters */
  const CalibratedTriad_<_T>& initGyroCalibration(){ return init_gyro_calib_; };
  
  /** @brief True if the accelerometers calibration is obtained using the mean
   *         accelerations of each static interval instead of all samples */
  bool accUseMeans() const { return acc_use_means_; };
  
  /** @brief Provides the (fixed) data period used in the gyroscopes integration. 
   *         If this period is less than 0, the gyroscopes timestamps are used
   *         in place of this period. */  
  _T gyroDataPeriod() const{ return gyro_dt_; };
  
  /** @brief True if the gyroscopes biases are estimated along with the calibration 
   *         parameters. If false, the gyroscopes biases (computed in the initial static
   *         period) are assumed known. */ 
  bool optimizeGyroBias() const { return optimize_gyro_bias_; };
  
  /** @brief True if the verbose output is enabled */ 
  bool verboseOutput() const { return verbose_output_; };
  
  /** @brief Set the magnitude of the gravitational filed 
   *         used in the calibration (i.e., the gravity measured in
   *         the place where the calibration dataset has been acquired)
   * 
   *         To find your magnitude of the gravitational filed, 
   *         take a look for example to https://www.wolframalpha.com
   */
  void setGravityMagnitude( _T g ){ g_mag_ = g; };
  
  /** @brief Set the duration in seconds of the initial static interval. Default 30 seconds. */
  _T setInitStaticIntervalDuration( _T duration ) { init_interval_duration_ = duration; };
  
  /** @brief Set the number of data samples to be extracted from each detected static intervals.
   *         Default is 100.  */
  int setIntarvalsNumSamples( int num ) { interval_n_samples_ = num; };
  
  /** @brief Set the accelerometers initial guess calibration parameters */  
  void setInitAccCalibration( CalibratedTriad_<_T> &init_calib ){ init_acc_calib_ = init_calib; };
  
  /** @brief Set the gyroscopes initial guess calibration parameters */
  void setInitGyroCalibration( CalibratedTriad_<_T> &init_calib ){ init_gyro_calib_ = init_calib; };
  
  /** @brief If the parameter enabled is true, the accelerometers calibration is obtained 
   *         using the mean accelerations of each static interval instead of all samples.
   *         Default is false.
   */
  void enableAccUseMeans ( bool enabled ){ acc_use_means_ = enabled; };
  
  /** @brief Set the (fixed) data period used in the gyroscopes integration. 
   *         If this period is less than 0, the gyroscopes timestamps are used
   *         in place of this period. Default is -1.
   */  
  void setGyroDataPeriod( _T dt ){ gyro_dt_ = dt; };
  
  /** @brief If the parameter enabled is true, the gyroscopes biases are estimated along
   *         with the calibration parameters. If false, the gyroscopes biases 
   *         (computed in the initial static period) are assumed known. */ 
  bool enableGyroBiasOptimization( bool enabled  ) { optimize_gyro_bias_ = enabled; };
  
  /** @brief If the parameter enabled is true, verbose output is activeted  */   
  void enableVerboseOutput( bool enabled ){ verbose_output_ = enabled; };
  
  /** @brief Estimate the calibration parameters for the acceleremoters triad 
   *         (see CalibratedTriad_) using the multi-position calibration method
   * 
   * @param acc_samples Acceleremoters data vector, ordered by increasing timestamps,
   *                    collected at the sensor data rate. 
   */
  bool calibrateAcc( const std::vector< TriadData_<_T> > &acc_samples );
  
  /** @brief Estimate the calibration parameters for both the acceleremoters 
   *         and the gyroscopes triads (see CalibratedTriad_) using the
   *         multi-position calibration method
   * 
   * @param acc_samples Acceleremoters data vector, ordered by increasing timestamps,
   *                    collected at the sensor data rate. 
   * @param gyro_samples Gyroscopes data vector, ordered by increasing timestamps,
   *                     collected in parallel with the acceleations 
   *                     at the sensor data rate.
   */
  bool calibrateAccGyro( const std::vector< TriadData_<_T> > &acc_samples, 
                         const std::vector< TriadData_<_T> > &gyro_samples );

  /** @brief Provide the calibration parameters for the acceleremoters triad (it should be called after
   *         calibrateAcc() or calibrateAccGyro() ) */
  const CalibratedTriad_<_T>& getAccCalib() const  { return acc_calib_; };
  /** @brief Provide the calibration parameters for the gyroscopes triad (it should be called after
   *         calibrateAccGyro() ). */
  const CalibratedTriad_<_T>& getGyroCalib() const  { return gyro_calib_; };
  
  /** @brief Provide the calibrated acceleremoters data vector (it should be called after
   *         calibrateAcc() or calibrateAccGyro() ) */
  const std::vector< TriadData_<_T> >& getCalibAccSamples() const { return calib_acc_samples_; };

  /** @brief Provide the calibrated gyroscopes data vector (it should be called after
   *         calibrateAccGyro() ) */
  const std::vector< TriadData_<_T> >& getCalibGyroSamples() const { return calib_gyro_samples_; };
  
private:
  
  _T g_mag_;
  const int min_num_intervals_;
  _T init_interval_duration_;
  int interval_n_samples_;
  bool acc_use_means_;
  _T gyro_dt_;
  bool optimize_gyro_bias_;
  std::vector< DataInterval > min_cost_static_intervals_;
  CalibratedTriad_<_T> init_acc_calib_, init_gyro_calib_;
  CalibratedTriad_<_T> acc_calib_, gyro_calib_;
  std::vector< TriadData_<_T> > calib_acc_samples_, calib_gyro_samples_;
  
  bool verbose_output_;
};

typedef MultiPosCalibration_<double> MultiPosCalibration;

}

/* Implementations */

template <typename _T> 
  imu_tk::CalibratedTriad_<_T>::CalibratedTriad_( const _T &mis_yz, const _T &mis_zy, const _T &mis_zx, 
                                                const _T &mis_xz, const _T &mis_xy, const _T &mis_yx, 
                                                const _T &s_x, const _T &s_y, const _T &s_z, 
                                                const _T &b_x, const _T &b_y, const _T &b_z )
{
  mis_mat_ <<  _T(1)   , -mis_yz  ,  mis_zy  ,
                mis_xz ,  _T(1)   , -mis_zx  ,  
               -mis_xy ,  mis_yx  ,  _T(1)   ;
              
  scale_mat_ <<   s_x  ,   _T(0)  ,  _T(0) ,
                 _T(0) ,    s_y   ,  _T(0) ,  
                 _T(0) ,   _T(0)  ,   s_z  ;
                    
  bias_vec_ <<  b_x , b_y , b_z ; 
  
  update();
}

template <typename _T> 
  bool imu_tk::CalibratedTriad_<_T>::load( std::string filename )
{
  std::ifstream file( filename.data() );
  if (file.is_open())
  {
    _T mat[9] = {0};
    
    for( int i=0; i<9; i++)
      file >> mat[i];

    mis_mat_ = Eigen::Map< const Eigen::Matrix< _T, 3, 3, Eigen::RowMajor> >(mat);
      
    for( int i=0; i<9; i++)
      file >> mat[i];
    
    scale_mat_ = Eigen::Map< const Eigen::Matrix< _T, 3, 3, Eigen::RowMajor> >(mat);
        
    for( int i=0; i<3; i++)
      file >> mat[i];
    
    bias_vec_ = Eigen::Map< const Eigen::Matrix< _T, 3, 1> >(mat);    
    
    update();
    
    return true;
  }
  return false;  
}

template <typename _T> 
  bool imu_tk::CalibratedTriad_<_T>::save( std::string filename ) const
{
  std::ofstream file( filename.data() );
  if (file.is_open())
  {
    file<<mis_mat_<<std::endl<<std::endl
        <<scale_mat_<<std::endl<<std::endl
        <<bias_vec_<<std::endl<<std::endl;
    
    return true;
  }
  return false;  
}

template <typename _T> void imu_tk::CalibratedTriad_<_T>::update()
{
  ms_mat_ = mis_mat_*scale_mat_;
}

template <typename _T> std::ostream& imu_tk::operator<<(std::ostream& os, 
                                                        const imu_tk::CalibratedTriad_<_T>& calib_triad)
{
  os<<"Misalignment Matrix"<<std::endl;
  os<<calib_triad.getMisalignmentMatrix()<<std::endl;
  os<<"Scale Matrix"<<std::endl;
  os<<calib_triad.getScaleMatrix()<<std::endl;
  os<<"Bias Vector"<<std::endl;
  os<<calib_triad.getBiasVector()<<std::endl;
  return os;
}