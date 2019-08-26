#include <iostream>

#include "imu_tk/io_utils.h"
#include "imu_tk/calibration.h"
#include "imu_tk/filters.h"
#include "imu_tk/integration.h"
#include "imu_tk/visualization.h"

using namespace std;
using namespace imu_tk;
using namespace Eigen;

int main(int argc, char** argv)
{
  if( argc < 3 )
    return -1;

  vector< TriadData > acc_data, gyro_data;
  
  cout<<"Importing IMU data from the Matlab matrix file : "<< argv[1]<<endl;  
  importAsciiData( argv[1], acc_data, imu_tk::TIMESTAMP_UNIT_SEC );
  cout<<"Importing IMU data from the Matlab matrix file : "<< argv[2]<<endl;  
  importAsciiData( argv[2], gyro_data, imu_tk::TIMESTAMP_UNIT_SEC  );
  
  
  CalibratedTriad init_acc_calib, init_gyro_calib;
  init_acc_calib.setBias( Vector3d(32768, 32768, 32768) );
  init_gyro_calib.setScale( Vector3d(1.0/6258.0, 1.0/6258.0, 1.0/6258.0) );
  
  MultiPosCalibration mp_calib;
    
  mp_calib.setInitStaticIntervalDuration(50.0);
  mp_calib.setInitAccCalibration( init_acc_calib );
  mp_calib.setInitGyroCalibration( init_gyro_calib );  
  mp_calib.setGravityMagnitude(9.81744);
  mp_calib.enableVerboseOutput(true);
  mp_calib.enableAccUseMeans(false);
  //mp_calib.setGyroDataPeriod(0.01);
  mp_calib.calibrateAccGyro(acc_data, gyro_data );
  mp_calib.getAccCalib().save("test_imu_acc.calib");
  mp_calib.getGyroCalib().save("test_imu_gyro.calib");
  
//   for( int i = 0; i < acc_data.size(); i++)
//   {
//     cout<<acc_data[i].timestamp()<<" "
  //         <<acc_data[i].x()<<" "<<acc_data[i].y()<<" "<<acc_data[i].z()<<" "
  //         <<gyro_data[i].x()<<" "<<gyro_data[i].y()<<" "<<gyro_data[i].z()<<endl;
//   }
//   cout<<"Read "<<acc_data.size()<<" tuples"<<endl;
  
  return 0;
}