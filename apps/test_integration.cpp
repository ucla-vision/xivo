#include <iostream>

#include <QApplication>

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
  if( argc < 2 )
    return -1;

  QApplication app(argc, argv);
  
  vector< TriadData > acc_data, gyro_data, mag_data;
  
  cout<<"Importing IMU data from file : "<< argv[1]<<endl;  
  importAsciiData( argv[1], acc_data, gyro_data, mag_data, 
                   imu_tk::TIMESTAMP_UNIT_USEC, imu_tk::DATASET_SPACE_SEPARATED);
  
  Vector3d gyro_bias = dataMean( gyro_data, DataInterval(100, 3000));
  CalibratedTriad bias_calib;
  bias_calib.setBias(gyro_bias);

  for(int i = 0; i < gyro_data.size(); i++)
    gyro_data[i] = bias_calib.unbias(gyro_data[i]);

  Vector3d mag_mean = dataMean( mag_data, DataInterval(100, 3000));
  
  Vis3D vis;
  
  Eigen::Vector4d quat(1.0, 0, 0, 0); // Identity quaternion
  double t[3] = {0, 0, 0};
  vis.registerFrame("ref");
  vis.registerFrame("cur", 255, 0, 0);
  vis.registerLine( "init_mag" );
  vis.registerLine( "mag" );
  
  vis.setFramePos( "ref", quat.data(), t );
  vis.setLinePos( "init_mag", t, mag_mean.data() );
  
  for(int i = 3000; i < gyro_data.size() - 1; i++)
  {
    double dt = gyro_data[i+1].timestamp() - gyro_data[i].timestamp();
    
    quatIntegrationStepRK4( quat, gyro_data[i].data(), gyro_data[i + 1].data(), dt, quat );
    vis.setFramePos( "cur", quat.data(), t );
    
    if( !(i%100) )
    {
      std::cout<<i/100<<std::endl;
    }
    if( !(i%5) )
    {
      vis.updateAndWait(1);
    }
  }

  std::cout<<"Done"<<std::endl;
  vis.updateAndWait();
  return 0;
}