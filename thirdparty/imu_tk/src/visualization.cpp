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

#include <limits>
#include <ceres/rotation.h>

#include "imu_tk/visualization.h"
#include "imu_tk/vis_extra/opengl_3d_scene.h"

#include <cstdio>
#include <sstream>

#include <QApplication>
#include <QKeyEvent>
#include <QTime>
#include <QString>

#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/concept_check.hpp>

using namespace imu_tk;

static int tmp_argc = 1;
static char *tmp_argv[] = { (char *)"" };

class Plot::PlotImpl
{
public:
  
  PlotImpl()
  { 
    gnuplot_pipe_ = popen("gnuplot", "w");
    if( gnuplot_pipe_ == NULL )
      std::cerr<<"WARNING : missing gnuplot application!"<<std::endl;
  };
  
  ~PlotImpl()
  { 
    if( gnuplot_pipe_ != NULL )
      pclose( gnuplot_pipe_ );
  };
  
  bool ready() const { return gnuplot_pipe_ != NULL; };
  void write( const std::string &str )
  {
    if( gnuplot_pipe_ != NULL )
    {
      fprintf( gnuplot_pipe_, "%s\n", str.c_str() );
      fflush( gnuplot_pipe_ );
    }
  };  
  
private:
  
  FILE *gnuplot_pipe_;
};

Plot::Plot()
{
  plot_impl_ptr_ = boost::shared_ptr< PlotImpl > ( new PlotImpl() );
}

template <typename _T> 
  void Plot::plotSamples ( const std::vector< TriadData_<_T> >& samples, 
                                   DataInterval range )
{
  if( !plot_impl_ptr_->ready() )
  {
    std::cerr<<"WARNING : can't plot samples: check that gnuplot is installed in your system!"<<std::endl;
    return;
  }
  
  range = checkInterval( samples, range );
  
  std::stringstream strs;
  strs<<"plot '-' title 'x' with lines, "
      <<"'-' title 'y' with lines, "
      <<"'-' title 'z' with lines"<<std::endl;
   
  _T base_time = samples[0].timestamp();
  for( int i = range.start_idx; i <= range.end_idx; i++)
    strs<<double(samples[i].timestamp() - base_time)<<" "<<double(samples[i].x())<<std::endl;
  strs<<"EOF"<<std::endl;
  for( int i = range.start_idx; i <= range.end_idx; i++)
    strs<<double(samples[i].timestamp() - base_time)<<" "<<double(samples[i].y())<<std::endl;
  strs<<"EOF"<<std::endl;
  for( int i = range.start_idx; i <= range.end_idx; i++)
    strs<<double(samples[i].timestamp() - base_time)<<" "<<double(samples[i].z())<<std::endl;
  strs<<"EOF"<<std::endl;
  plot_impl_ptr_->write( strs.str() );
}

template <typename _T> 
  void Plot::plotIntervals ( const std::vector< TriadData_< _T > >& samples,
                                     const std::vector< DataInterval >& intervals, 
                                     DataInterval range )
{
  if( !plot_impl_ptr_->ready() )
  {
    std::cerr<<"WARNING : can't plot samples: check that gnuplot is installed in your system!"<<std::endl;
    return;
  }
   
  range = checkInterval( samples, range );
  int n_pts = range.end_idx - range.start_idx + 1, 
              n_intervals = intervals.size();
  
  double max = 0, mean = 0;
  for( int i = range.start_idx; i <= range.end_idx; i++)
  {
    if( double(samples[i].x()) > max ) max = double(samples[i].x());
    if( double(samples[i].y()) > max ) max = double(samples[i].y());
    if( double(samples[i].z()) > max ) max = double(samples[i].z());
    
    mean += (double(samples[i].x()) + double(samples[i].y()) + double(samples[i].z()))/3;
  }
  
  mean /= n_pts;
  max -= mean;
  double step_h = mean + max/2, val = 0;
  int interval_idx = 0;
  for( ; interval_idx < n_intervals; interval_idx++ )
  {
    if (intervals[interval_idx].start_idx >= range.start_idx )
      break;
  }
  
  std::stringstream strs;
  strs<<"plot '-' title 'x' with lines, "
      <<"'-' title 'y' with lines, "
      <<"'-' title 'z' with lines, "
      <<"'-' title 'intervals' with lines"<<std::endl;
   
  _T base_time = samples[0].timestamp();
  for( int i = range.start_idx; i <= range.end_idx; i++)
    strs<<double(samples[i].timestamp() - base_time)<<" "<<double(samples[i].x())<<std::endl;
  strs<<"EOF"<<std::endl;
  for( int i = range.start_idx; i <= range.end_idx; i++)
    strs<<double(samples[i].timestamp() - base_time)<<" "<<double(samples[i].y())<<std::endl;
  strs<<"EOF"<<std::endl;
  for( int i = range.start_idx; i <= range.end_idx; i++)
    strs<<double(samples[i].timestamp() - base_time)<<" "<<double(samples[i].z())<<std::endl;
  strs<<"EOF"<<std::endl;
  
  for( int i = range.start_idx; i <= range.end_idx; i++)
  {
    if( interval_idx < n_intervals)
    {
      if( i == intervals[interval_idx].start_idx )
        val = step_h;
      else if( i == intervals[interval_idx].end_idx)
      {
        val = 0;
        interval_idx++;
      }
    }
    strs<<double(samples[i].timestamp() - base_time)<<" "<<val<<std::endl;
  }
  strs<<"EOF"<<std::endl;
  
  plot_impl_ptr_->write( strs.str() );
}

void imu_tk::waitForKey()
{
  do
  {
    std::cout<<std::endl<<"Press Enter to continue"<<std::endl;
  }
  while ( getchar() != '\n' );  
}

class Vis3D::VisualizerImpl : public OpenGL3DScene
{
public  :
  VisualizerImpl( QWidget * parent = 0, Qt::WindowFlags f = 0 ) :
    waiting_for_key(false)
  { 
    if ( !QApplication::instance() )
    {
      new QApplication( tmp_argc, tmp_argv );
    }

    moveToThread(QApplication::instance()->thread());

    setAutoAdjust(false);
    setFirstPersonView(false);
    moveCamera(-1.5f, -1.5f, 0.5f, 90.0f, -45.0f,0.0f);
    setCameraIncrements(0.1,2.0);
    resize( 640,480 );
    show();
    
  };
  ~VisualizerImpl() {};
  // TODO Ugly workaround
  bool waiting_for_key;
  
protected:

  virtual void keyPressEvent ( QKeyEvent * event )
  {
    OpenGL3DScene::keyPressEvent ( event );
    if( event->key() == Qt::Key_Escape )
      waiting_for_key = false;
  };
};


Vis3D::Vis3D ( const std::string win_name )
{
  vis_impl_ptr_ = boost::shared_ptr< VisualizerImpl > ( new VisualizerImpl() );
  QString w_name( win_name.c_str() );
  w_name += " - press h for help";
  vis_impl_ptr_->setWindowTitle(w_name);
}

void Vis3D::registerFrame( std::string name, uint8_t r, uint8_t g, uint8_t b )
{
  vis_impl_ptr_->registerAxes(name, QColor(r,g,b) );
}

void Vis3D::unregisterFrame( std::string name )
{
  vis_impl_ptr_->unregisterAxes(name);
}
    
template <typename _T> 
  void Vis3D::setFramePos( std::string name, const _T quat[4], const _T t[3] )
{
  Eigen::Vector4d q_vec;
  Eigen::Vector3d r_vec, t_vec;
  q_vec<<quat[0], quat[1], quat[2], quat[3];
  t_vec<<t[0], t[1], t[2];
  ceres::QuaternionToAngleAxis( q_vec.data(), r_vec.data() );
  vis_impl_ptr_->setAxesPos(name, r_vec, t_vec );
}


void Vis3D::registerLine( std::string name, uint8_t r, uint8_t g, uint8_t b )
{
  vis_impl_ptr_->registerLine(name, QColor(r,g,b));
}

void Vis3D::unregisterLine( std::string name )
{
  vis_impl_ptr_->unregisterLine(name);
}

template <typename _T> 
  void Vis3D::setLinePos( std::string name, const _T p0[3], const _T p1[3] )
{
  Eigen::Vector3d p0_vec, p1_vec;
  p0_vec<<p0[0], p0[1], p0[2];
  p1_vec<<p1[0], p1[1], p1[2];

  vis_impl_ptr_->setLine(name, p0_vec, p1_vec);
}

void Vis3D::updateAndWait( int delay_ms )
{
  vis_impl_ptr_->waiting_for_key = true;
  QTime time;
  
  if( delay_ms > 0 )
    time.start();
  
  while( vis_impl_ptr_->waiting_for_key )
  {
    if ( !QApplication::instance() )
    {
      new QApplication( tmp_argc, tmp_argv );
    }
    vis_impl_ptr_->updateNow();
    QApplication::instance()->processEvents();
      
    usleep(1000);
    
    // TODO Improve here using a timer
    if( delay_ms > 0 && time.elapsed() > delay_ms )
      vis_impl_ptr_->waiting_for_key = false;
  }
}

template void Plot::plotSamples<double> ( const std::vector< TriadData_<double> >& samples, 
                                          DataInterval range );
template void Plot::plotSamples<float> ( const std::vector< TriadData_<float> >& samples, 
                                         DataInterval range );
template void Plot::plotIntervals<double> ( const std::vector< TriadData_<double> >& samples, 
                                            const std::vector< DataInterval >& intervals,
                                            DataInterval range );
template void Plot::plotIntervals<float> ( const std::vector< TriadData_<float> >& samples, 
                                           const std::vector< DataInterval >& intervals,
                                           DataInterval range );

template void Vis3D::setFramePos<double>( std::string name, const double quat[4], const double t[3] );
template void Vis3D::setFramePos<float>( std::string name, const float quat[4], const float t[3] );
template void Vis3D::setLinePos<double>( std::string name, const double p0[3], const double p1[3] );
template void Vis3D::setLinePos<float>( std::string name, const float p0[3], const float p1[3] );
