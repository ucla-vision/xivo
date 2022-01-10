#include <iostream>
#include <fstream>

#include "imu_tk/imu_tk.h"

using namespace std;
using namespace imu_tk;
using namespace Eigen;

int main(int argc, char** argv)
{
  if( argc < 3 )
    return -1;

  int n_intervals = 1;
  if( argc >= 4)
    n_intervals = atoi(argv[3]);
    
  vector< TriadData > acc_data_calib, gyro_data_calib;
  
  cout<<"Importing IMU data from the comma separated file : "<< argv[1]<<endl;  
  importAsciiData( argv[1], acc_data_calib, gyro_data_calib, 
                   imu_tk::TIMESTAMP_UNIT_SEC, imu_tk::DATASET_COMMA_SEPARATED );
  
  MultiPosCalibration mp_calib;
    
  mp_calib.setInitStaticIntervalDuration(50.0);
  mp_calib.setGravityMagnitude(9.803);
  mp_calib.enableVerboseOutput(false);
  mp_calib.enableAccUseMeans(false);
  //mp_calib.setGyroDataPeriod(0.01);
  mp_calib.calibrateAccGyro(acc_data_calib, gyro_data_calib );
//   mp_calib.getAccCalib().save("test_phone_acc.calib");
//   mp_calib.getGyroCalib().save("test_phone_gyro.calib");
  
  vector< TriadData > acc_data_test, gyro_data_test, gyro_data_test_calib;
  
  cout<<"Importing IMU data from the comma separated file : "<< argv[2]<<endl;  
  importAsciiData( argv[2], acc_data_test, gyro_data_test, 
                   imu_tk::TIMESTAMP_UNIT_SEC, imu_tk::DATASET_COMMA_SEPARATED );
  
  for(int i = 0; i < gyro_data_test.size(); i++)
    gyro_data_test_calib.push_back(mp_calib.getGyroCalib().normalize(gyro_data_test[i]));
  

  vector<DataInterval>tmp_intervals, static_intervals;
  vector<TriadData >acc_means;
  Vector3d variance = dataVariance( acc_data_test, DataInterval(100, 3000));
  staticIntervalsDetector( acc_data_test, 4*variance.norm(), tmp_intervals);
  extractIntervalsSamples( acc_data_test, tmp_intervals, acc_means, static_intervals, 100, true);
  
  
  Plot plot1, plot2;
  plot1.plotIntervals(gyro_data_test, static_intervals);
//   plot1.plotSamples( gyro_data_test);
  
  Vector3d gyro_bias = dataMean( gyro_data_test_calib, DataInterval(100, 3000));
  CalibratedTriad bias_calib;
  bias_calib.setBias(gyro_bias);

  for(int i = 0; i < gyro_data_test_calib.size(); i++)
    gyro_data_test_calib[i] = bias_calib.unbias(gyro_data_test_calib[i]);
  
  gyro_bias = dataMean( gyro_data_test, DataInterval(100, 3000));
  bias_calib.setBias(gyro_bias);

  for(int i = 0; i < gyro_data_test.size(); i++)
    gyro_data_test[i] = bias_calib.unbias(gyro_data_test[i]);
  
  
//   plot1.plotSamples( gyro_data_test);
//   waitForKey();


  std::ofstream file( "results.mat", ios_base::app );
  if (!file.is_open())
  {
    return -1;
  }
//   
  int uncalib_score = 0, calib_score = 0;
  cout<<"n_intervals :"<<n_intervals<<endl;
  for(int i = n_intervals; i < static_intervals.size(); i++)
  {
    Vector3d res, res_calib;
    Matrix3d test_rot_res, test_rot_res_calib;
    
    integrateGyroInterval( gyro_data_test, test_rot_res, double(-1), DataInterval(static_intervals[i-n_intervals].end_idx, static_intervals[i].start_idx) );
    integrateGyroInterval( gyro_data_test_calib, test_rot_res_calib, double(-1), DataInterval(static_intervals[i-n_intervals].end_idx, static_intervals[i].start_idx) );
    
    decomposeRotation(test_rot_res, res);
    decomposeRotation(test_rot_res_calib, res_calib);
    
    res *= 180/M_PI;
    res_calib *= 180/M_PI;
    double ts0 =  acc_data_test[static_intervals[i-n_intervals].end_idx].timestamp(),
           ts1 = acc_data_test[static_intervals[i].start_idx].timestamp();
    
    double interval_len = ts1 - ts0;
    cout<<"Interval len:"<<interval_len
        <<" from "<<ts0 - gyro_data_test[0].timestamp()
        <<" to "<<ts1 - gyro_data_test[0].timestamp()<<endl;
    cout<<"Uncalib : "<<res.norm() <<endl;
    cout<<"Calib   : "<<res_calib.norm()<<endl<<endl;

    file<<interval_len<<" "<<res.norm()<<" "<<res_calib.norm()<<endl;
//     if(res.norm() - res_calib(2)) > 0.01)
//     {
    if(res.norm() <= res_calib.norm())
      uncalib_score++;
    else
      calib_score++;
//     }
  }
  cout<<"Uncalib : "<<uncalib_score<<endl;
  cout<<"Calib : "<<calib_score<<endl;  
//   integrateGyroInterval( gyro_data_calib, calib_rot_res );
//   integrateGyroInterval( mp_calib.getCalibGyroSamples(), calib_rot_res_calib );
// 
//   cout<<calib_rot_res*test_vec <<endl<<endl;
//   cout<<calib_rot_res_calib*test_vec<<endl<<endl;
  
//   cout<<test_rot_res<<endl;
//   cout<<calib_rot_res<<endl;
  
  
//   for( int i = 0; i < acc_data.size(); i++)
//   {
//     cout<<acc_data[i].timestamp()<<" "
  //         <<acc_data[i].x()<<" "<<acc_data[i].y()<<" "<<acc_data[i].z()<<" "
  //         <<gyro_data[i].x()<<" "<<gyro_data[i].y()<<" "<<gyro_data[i].z()<<endl;
//   }
//   cout<<"Read "<<acc_data.size()<<" tuples"<<endl;
  
   waitForKey();
  return 0;
}