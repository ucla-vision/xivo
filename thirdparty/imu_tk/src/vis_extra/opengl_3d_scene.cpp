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

#include <QtOpenGL>
#include <QMutexLocker>

#include <GL/glut.h>

#include <typeinfo>
#include <math.h>
#include <boost/iterator/iterator_concepts.hpp>

#include "imu_tk/vis_extra/opengl_3d_scene.h"

#define OPENGL3DSCENE_DRAW_AXIS_TICKS(val) { \
      glVertex3d(val, -0.02f, 0.0f);\
      glVertex3d(val, 0.02f, 0.0f);\
      glVertex3d(val, 0.0f, -0.02f);\
      glVertex3d(val, 0.0f, 0.012);\
      glVertex3d(-0.02f, val, 0.0f);\
      glVertex3d(0.02f, val, 0.0f);\
      glVertex3d(0.0f, val, -0.02f);\
      glVertex3d(0.0f, val, 0.02f);\
      glVertex3d(-0.01f, 0.0f, val);\
      glVertex3d(0.02f, 0.0f, val);\
      glVertex3d(0.0f, -0.02f, val);\
      glVertex3d(0.0f, 0.02f, val); }

OpenGL3DScene::OpenGL3DScene ( QWidget* parent, QColor background_color, 
                               bool global_frame, QColor global_frame_color )
  : QGLWidget ( parent ), 
  mutex_ ( QMutex::Recursive ), 
  bg_color_ ( background_color ), 
  left_mouse_button_pressed_(false),
  right_mouse_button_pressed_(false),
  mouse_init_x_(0), 
  mouse_init_y_(0)
{
  fp_view_ = false;
  auto_adjust_ = true;
  auto_adjust_offset_ = 20.0;

  view_angle_ = GLfloat ( M_PI/3.0 );
  form_factor_ = 1.0f;
  max_x_ = max_y_= -std::numeric_limits< float >::max();
  min_x_ = min_y_ =std::numeric_limits< float >::min();

  t_inc_ = 0.6f;
  r_inc_ = 5.0f;

  makeCurrent();
  axis_list_ = glGenLists ( 1 );
  enableGlobalAxes ( global_frame, global_frame_color );
  camera_.reset();

  setFocusPolicy ( Qt::StrongFocus );
  
  connect( this, SIGNAL( sceneUpdated() ), this, SLOT( updateAllStructure() ));
  //setFocusPolicy ( Qt::NoFocus );//WARNING debug!!
}

OpenGL3DScene::~OpenGL3DScene()
{
  unregisterAllStructures();
}

void OpenGL3DScene::enableGlobalAxes ( bool enable, QColor color )
{
  draw_axes_ = enable;

  if ( enable )
  {
    makeCurrent();
    glNewList ( axis_list_,GL_COMPILE );

    qglColor ( color );
    GLfloat axis_limit = 1.0f;

    glBegin ( GL_LINES );

    glVertex3d ( -axis_limit, 0.0f, 0.0f );
    glVertex3d ( axis_limit, 0.0f, 0.0f );

    glVertex3d ( 0.0f, -axis_limit, 0.0f );
    glVertex3d ( 0.0f, axis_limit, 0.0f );

    glVertex3d ( 0.0f, 0.0f, -axis_limit );
    glVertex3d ( 0.0f, 0.0f, axis_limit );

    for ( GLfloat inc = -1.0f; inc <= 1.0f; inc +=0.1f )
      OPENGL3DSCENE_DRAW_AXIS_TICKS ( inc );

    for ( GLfloat inc = -axis_limit; inc <= axis_limit; inc +=1.0f )
      OPENGL3DSCENE_DRAW_AXIS_TICKS ( inc );

    glEnd();

    glEndList();
  }
}

void OpenGL3DScene::resizeGL ( int width, int height )
{
  /* Specifies the affine transformation of x and y
   from normalized device coordinates to window coordinate */
  glViewport ( 0, 0, ( GLint ) width, ( GLint ) height );

  int side = qMin ( width, height );
  form_factor_ = GLfloat ( width ) /GLfloat ( height );

  GLfloat clipping_planes_hside = GLfloat ( tanf ( view_angle_/2.0f ) );

  glMatrixMode ( GL_PROJECTION );
  glLoadIdentity();

  glFrustum ( -clipping_planes_hside*form_factor_, clipping_planes_hside*form_factor_,
              -clipping_planes_hside, clipping_planes_hside, 0.5, 2001.0 );

  glMatrixMode ( GL_MODELVIEW );

}

void OpenGL3DScene::initializeGL()
{
  //glClearColor(0.0, 0.0, 0.0, 0.0);
  qglClearColor ( bg_color_ );
  //glShadeModel(GL_FLAT);
  glShadeModel ( GL_SMOOTH );
  glEnable ( GL_DEPTH_TEST );
  glEnable ( GL_CULL_FACE );
  glDisable ( GL_LIGHTING );
  glEnable ( GL_NORMALIZE );

  GLfloat mat_shininess[] = { 50.0 };
  GLfloat light_position[] = { 100.0, 100.0, 100.0, 0.0 };
  GLfloat model_ambient[] = { 0.5, 0.5, 0.5, 1.0 };

  glMaterialfv ( GL_FRONT, GL_SHININESS, mat_shininess );
  glLightfv ( GL_LIGHT0, GL_POSITION, light_position );
  glLightModelfv ( GL_LIGHT_MODEL_AMBIENT, model_ambient );

  glEnable ( GL_LIGHT0 );

}

void OpenGL3DScene::paintGL()
{
  glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

  glLoadIdentity();
  camera_.render();
  // Call all object lists

  if ( draw_axes_ )
    glCallList ( axis_list_ );


  std::vector< GLuint > lists;
  
  std::map <std::string, PathItem>::iterator path_iter;
  for ( path_iter = path_items_.begin(); path_iter != path_items_.end(); path_iter++ )
  {
    lists.insert(lists.end(), path_iter->second.lists.begin(), path_iter->second.lists.end());
  }

  std::map <std::string, CloudItem>::iterator cloud_iter;
  for ( cloud_iter = cloud_items_.begin(); cloud_iter != cloud_items_.end(); cloud_iter++ )
  {
    lists.insert(lists.end(), cloud_iter->second.lists.begin(), cloud_iter->second.lists.end());
  }

  
  if( lists.size() )
  {
    // Execute the list of display lists
    glCallLists ( lists.size(), GL_UNSIGNED_INT, ( const GLvoid * ) lists.data() );
  }
  
  
  std::map <std::string, AxesItem>::iterator axes_iter;
  for ( axes_iter = axes_items_.begin(); axes_iter != axes_items_.end(); axes_iter++ )
  {
    if ( axes_iter->second.list != GL_INVALID_VALUE )
    {
      glCallList ( axes_iter->second.list );
    }
  }
}

void OpenGL3DScene::setZoom ( int z )
{

//   if( !fp_view )
//   {
//     zT = float(z);
//     updateGL();
//   }

}

void OpenGL3DScene::updateNow()
{
  sceneUpdated();
}

void OpenGL3DScene::updateStructure( std::string name, bool update_gl )
{
  makeCurrent();
    
  std::map <std::string, PathItem>::iterator path_iter;
  if( (path_iter = path_items_.find(name)) != path_items_.end() )
  {
    PathItem &p_item = path_iter->second;
    if( p_item.removed ) 
    {
      for( int i = 0; i < path_iter->second.lists.size(); i++)
        glDeleteLists ( path_iter->second.lists[i], 1 );
      path_items_.erase(path_iter);
      
    }
    else
    {
      // Create a display list
      GLuint new_list = glGenLists ( 1 );
      p_item.lists.push_back(new_list);
      glNewList ( new_list, GL_COMPILE );
      qglColor ( p_item.color );

      for( int i = 0; i < int(p_item.unprocessed_poses.size()) - 1; i++ )      
      {
        Eigen::Vector3d &p0 = p_item.unprocessed_poses[i], &p1 = p_item.unprocessed_poses[i+1];
        
        // Update 2D boundaries (X,Y) of the current drawn scene
        if ( p0(0) > max_x_ ) max_x_ = p0(0);
        if ( p0(0) < min_x_ ) min_x_ = p0(0);
        if ( p0(1) > max_y_ ) max_y_ = p0(1);
        if ( p0(1) < min_y_ ) min_y_ = p0(1);

        if ( p1(0) > max_x_ ) max_x_ = p1(0);
        if ( p1(0) < min_x_ ) min_x_ = p1(0);
        if ( p1(1) > max_y_ ) max_y_ = p1(1);
        if ( p1(1) < min_y_ ) min_y_ = p1(1);
      
        glBegin ( GL_LINES );

        glVertex3d ( p0(0), p0(1), p0(2) );
        glVertex3d ( p1(0), p1(1), p1(2) );

        glEnd();
      }

      glEndList();

      Eigen::Vector3d last_pos = p_item.unprocessed_poses.back();
      p_item.unprocessed_poses.clear();
      p_item.unprocessed_poses.push_back(last_pos);
    }
  }
  
  std::map <std::string, LineItem>::iterator lines_iter;  
  if( (lines_iter = line_items_.find(name)) != line_items_.end()  ) 
  {
    LineItem &l_item = lines_iter->second;
    
    if(l_item.removed)
    {
      if ( lines_iter->second.list != GL_INVALID_VALUE )
        glDeleteLists ( lines_iter->second.list, 1 );
      line_items_.erase(lines_iter);
    }  
    else if( l_item.updated )
    {
      if ( l_item.list != GL_INVALID_VALUE )
        glDeleteLists ( l_item.list, 1 );
      
      Eigen::Vector3d &p0 = l_item.p0, &p1 = l_item.p1;
      
      l_item.list = glGenLists ( 1 );
      
      glNewList ( l_item.list, GL_COMPILE );
      qglColor ( l_item.color );

      glBegin ( GL_LINES );

      glVertex3d ( p0(0), p0(1), p0(2) );
      glVertex3d ( p1(0), p1(1), p1(2) );

      glEnd();
      
      glEndList();
    }
  }
  
  std::map <std::string, CloudItem>::iterator cloud_iter; 
  if( (cloud_iter = cloud_items_.find(name)) != cloud_items_.end() )
  {
    CloudItem &c_item = cloud_iter->second;

    if( c_item.removed )
    {
      for( int i = 0; i < cloud_iter->second.lists.size(); i++)
        glDeleteLists ( cloud_iter->second.lists[i], 1 );
      cloud_items_.erase(cloud_iter);
    }
    else
    {
      // Create a display list
      GLuint new_list = glGenLists ( 1 );
      c_item.lists.push_back(new_list);
      glNewList ( new_list, GL_COMPILE );
      qglColor ( c_item.color );

      glBegin ( GL_POINTS );
      for( int i = 0; i < int(c_item.unprcessed_cloud.size()); i++ )      
      {
        Eigen::Vector3d &p = c_item.unprcessed_cloud[i];
        glVertex3d ( GLfloat(p(0)), GLfloat(p(1)), GLfloat(p(2)) );
      }
      glEnd();
      glEndList();

      c_item.unprcessed_cloud.clear();
    }
  }
  
  std::map <std::string, AxesItem>::iterator axes_iter;  
  if( (axes_iter = axes_items_.find(name)) != axes_items_.end()  ) 
  {
    AxesItem &a_item = axes_iter->second;
    
    if(a_item.removed)
    {
      if ( axes_iter->second.list != GL_INVALID_VALUE )
        glDeleteLists ( axes_iter->second.list, 1 );
      for ( int i = 0; i < 7; i++ )
        gluDeleteQuadric ( axes_iter->second.qobjs[i] );
      axes_items_.erase(axes_iter);
    }  
    else if( a_item.updated )
    {
      if ( a_item.list != GL_INVALID_VALUE )
        glDeleteLists ( a_item.list, 1 );
      
      a_item.list = glGenLists ( 1 );
      
      glNewList ( a_item.list, GL_COMPILE );

      glEnable ( GL_LIGHTING );

      qglColor ( a_item.color );

      GLfloat mat_ambient[] = { a_item.color.redF(), a_item.color.greenF(), a_item.color.blueF(), 1.0f };
      GLfloat mat_diffuse[] = { a_item.color.redF() /2.0f, a_item.color.greenF() /2.0f,
                                a_item.color.blueF() /2.0f, 1.0f };
      glMaterialfv ( GL_FRONT, GL_AMBIENT, mat_ambient );
      glMaterialfv ( GL_FRONT, GL_DIFFUSE, mat_diffuse );

      glPushMatrix();

      glTranslatef ( a_item.t(0), a_item.t(1), a_item.t(2) );

      glRotatef ( 180.0*sqrt ( ( a_item.r(0)*a_item.r(0) ) + ( a_item.r(1)*a_item.r(1) ) + ( a_item.r(2)*a_item.r(2) ) ) /M_PI,
                  a_item.r(0), a_item.r(1), a_item.r(2) );


      // Center
      gluSphere ( a_item.qobjs[0], 0.02, 5, 5 );

      // Z-Axis
      gluCylinder ( a_item.qobjs[1], 0.02, 0.02, 0.5, 10, 5 );
      glPushMatrix();
      glTranslatef ( 0.0, 0.0, 0.45 );

      mat_ambient[0] = 1.0f - mat_ambient[0];
      mat_ambient[1] = 1.0f - mat_ambient[1];
      mat_ambient[2] = 1.0f - mat_ambient[2];
      mat_diffuse[0] = 1.0f - mat_diffuse[0];
      mat_diffuse[1] = 1.0f - mat_diffuse[1];
      mat_diffuse[2] = 1.0f - mat_diffuse[2];

      glMaterialfv ( GL_FRONT, GL_AMBIENT, mat_ambient );
      glMaterialfv ( GL_FRONT, GL_DIFFUSE, mat_diffuse );

      gluCylinder ( a_item.qobjs[2], 0.04, 0.0, 0.1, 15, 5 );

      mat_ambient[0] = 1.0f - mat_ambient[0];
      mat_ambient[1] = 1.0f - mat_ambient[1];
      mat_ambient[2] = 1.0f - mat_ambient[2];
      mat_diffuse[0] = 1.0f - mat_diffuse[0];
      mat_diffuse[1] = 1.0f - mat_diffuse[1];
      mat_diffuse[2] = 1.0f - mat_diffuse[2];
      glMaterialfv ( GL_FRONT, GL_AMBIENT, mat_ambient );
      glMaterialfv ( GL_FRONT, GL_DIFFUSE, mat_diffuse );

      glPopMatrix();

      // Y-Axis
      glPushMatrix();
      glRotatef ( -90.0,1.0,0, 0 );
      gluCylinder ( a_item.qobjs[3], 0.02, 0.02, 0.5, 10, 5 );
      glPushMatrix();
      glTranslatef ( 0.0, 0.0, 0.45 );
      gluCylinder ( a_item.qobjs[4], 0.04, 0.0, 0.1, 15, 5 );
      glPopMatrix();
      glPopMatrix();

      // X-Axis
      glPushMatrix();
      glRotatef ( 90.0,0.0,1.0, 0 );
      gluCylinder ( a_item.qobjs[5], 0.02, 0.02, 0.5, 10, 5 );
      glPushMatrix();
      glTranslatef ( 0.0, 0.0, 0.45 );
      gluCylinder ( a_item.qobjs[6], 0.04, 0.0, 0.1, 15, 5 );
      glPopMatrix();
      glPopMatrix();

      glPopMatrix();

      glDisable ( GL_LIGHTING );
      glEndList();
    }
  }

  if ( auto_adjust_ )
    autoAdjustGView();

  if(update_gl)
    updateGL();
}

void OpenGL3DScene::moveCamera ( GLfloat tx, GLfloat ty, GLfloat tz, GLfloat r1x, GLfloat r2y, GLfloat r3z )
{
  camera_.reset();

  camera_.reset();
  camera_.moveAbs ( tx, ty, tz );
  camera_.rotateX ( r1x );
  camera_.rotateY ( r2y );
  camera_.rotateZ ( r3z );

  updateGL();
}

void OpenGL3DScene::setFirstPersonView ( bool good_pos )
{
  fp_view_ = true;

  if ( good_pos )
    moveCamera ( 0.0f, 0.0f, 1.8f, 90.0, 0.0f, 0.0f );
  updateGL();
}

void OpenGL3DScene::setGlobalView()
{
  fp_view_ = false;

  camera_.reset();
  autoAdjustGView();

  updateGL();
}

void OpenGL3DScene::autoAdjustGView()
{

  makeCurrent();
  GLfloat center_x = 0.0f, center_y = 0.0f;
  GLfloat x_t = 0.0f, y_t = 0.0f, z_t = 1.0f;

  if ( max_x_ != -std::numeric_limits< float >::max() && max_y_ != -std::numeric_limits< float >::max() &&
       min_x_ != std::numeric_limits< float >::min() && min_y_ != std::numeric_limits< float >::min() )
  {
    center_x = min_x_ + ( max_x_ - min_x_ ) / 2.0f;
    center_y = min_y_ + ( max_y_ - min_y_ ) / 2.0f;

    x_t = center_x;
    y_t = center_y;

    GLfloat half_width = qMax ( center_x - min_x_,max_x_ - center_x ) + auto_adjust_offset_;
    GLfloat half_height = qMax ( center_y - min_y_,max_y_ - center_y ) + auto_adjust_offset_;

    GLfloat half_side = qMax ( half_width,half_height );

    GLfloat z_t = half_side/tanf ( view_angle_/2.0 );

    if ( z_t < 1.0f ) z_t = 1.0f;

    camera_.moveAbs ( x_t, y_t, z_t );

  }
//  emit zoomChanged(zT);
}


void OpenGL3DScene::registerPath( std::string name, QColor path_color )
{
  QMutexLocker locker ( &mutex_ );
  
  if( path_items_.find(name) != path_items_.end() )
    return;
  
  PathItem new_path;
  new_path.color = path_color;
  path_items_[name] = new_path;
}

void OpenGL3DScene::unregisterPath( std::string name )
{
  QMutexLocker locker ( &mutex_ );
  
  std::map <std::string, PathItem>::iterator path_iter = path_items_.find(name);
  if( path_iter != path_items_.end() )
    path_iter->second.removed = true;
}

void OpenGL3DScene::pushPosition( std::string name, const Eigen::Vector3d& t )
{
  QMutexLocker locker ( &mutex_ );
  
  if( path_items_.find(name) == path_items_.end() )
    return;
  path_items_[name].unprocessed_poses.push_back(t);
}

void OpenGL3DScene::registerLine( std::string name, QColor line_color )
{
  QMutexLocker locker ( &mutex_ );
  
  if( line_items_.find(name) != line_items_.end() )
    return;
  
  LineItem new_line;
  new_line.color = line_color;
  new_line.list = GL_INVALID_VALUE;
  line_items_[name] = new_line;
}

void OpenGL3DScene::unregisterLine( std::string name )
{
  QMutexLocker locker ( &mutex_ );

  std::map <std::string, LineItem>::iterator lines_iter = line_items_.find(name);
  if( lines_iter != line_items_.end() )
    lines_iter->second.removed = true;
}

void OpenGL3DScene::setLine( std::string name, const Eigen::Vector3d& p0, const Eigen::Vector3d& p1 )
{
  QMutexLocker locker ( &mutex_ );
  
  if( line_items_.find(name) == line_items_.end() )
    return;  
  
  line_items_[name].p0 = p0;
  line_items_[name].p1 = p1;
  line_items_[name].updated = true;
}

void OpenGL3DScene::registerCloud( std::string name, QColor cloud_color )
{
  QMutexLocker locker ( &mutex_ );
  
  if( cloud_items_.find(name) != cloud_items_.end() )
    return;
  
  CloudItem new_cloud;
  new_cloud.color = cloud_color;
  cloud_items_[name] = new_cloud;
}

    
void OpenGL3DScene::unregisterCloud( std::string name )
{

  QMutexLocker locker ( &mutex_ );
  
  std::map <std::string, CloudItem>::iterator cloud_iter = cloud_items_.find(name);
  if( cloud_iter != cloud_items_.end() )
    cloud_iter->second.removed = true;
}

void OpenGL3DScene::pushCloud( std::string name, const std::vector< Eigen::Vector3d > &pts )
{
  QMutexLocker locker ( &mutex_ );
  if( cloud_items_.find(name) == cloud_items_.end() )
    return;

  cloud_items_[name].unprcessed_cloud.insert(cloud_items_[name].unprcessed_cloud.end(), 
                                             pts.begin(), pts.end());
}

void OpenGL3DScene::registerAxes( std::string name, QColor axes_color )
{
  QMutexLocker locker ( &mutex_ );

  if( axes_items_.find(name) != axes_items_.end() )
    return;
  
  AxesItem new_axes;
  new_axes.color = axes_color;
  new_axes.list = GL_INVALID_VALUE;
  
  for ( int i = 0; i < 7; i++ )
  {
    new_axes.qobjs[i] = gluNewQuadric();
    gluQuadricNormals ( new_axes.qobjs[i], GLU_SMOOTH );
  }
  
  axes_items_[name] = new_axes;
}

void OpenGL3DScene::unregisterAxes( std::string name )
{
  QMutexLocker locker ( &mutex_ );

  std::map <std::string, AxesItem>::iterator axes_iter = axes_items_.find(name);
  if( axes_iter != axes_items_.end() )
    axes_iter->second.removed = true;
}

void OpenGL3DScene::setAxesPos( std::string name, const Eigen::Vector3d& r, const Eigen::Vector3d& t )
{
  QMutexLocker locker ( &mutex_ );
  
  if( axes_items_.find(name) == axes_items_.end() )
    return;  
  
  axes_items_[name].r = r;
  axes_items_[name].t = t;
  axes_items_[name].updated = true;
}

void OpenGL3DScene::unregisterAllStructures()
{
  // WARNING MAYBE BUG
  QMutexLocker locker ( &mutex_ );
  
  std::map <std::string, PathItem>::iterator path_iter;
  for ( path_iter = path_items_.begin(); path_iter != path_items_.end(); path_iter++ )
  {
    for( int i = 0; i < path_iter->second.lists.size(); i++)
      glDeleteLists ( path_iter->second.lists[i], 1 );
  }
  
  path_items_.clear();

  std::map <std::string, LineItem>::iterator lines_iter;
  for ( lines_iter = line_items_.begin(); lines_iter != line_items_.end(); lines_iter++ )
  {
    if ( lines_iter->second.list != GL_INVALID_VALUE )
      glDeleteLists ( lines_iter->second.list, 1 );
  }
  
  line_items_.clear();
  
  std::map <std::string, CloudItem>::iterator cloud_iter;
  for ( cloud_iter = cloud_items_.begin(); cloud_iter != cloud_items_.end(); cloud_iter++ )
  {
    for( int i = 0; i < cloud_iter->second.lists.size(); i++)
      glDeleteLists ( cloud_iter->second.lists[i], 1 );
  }
  
  cloud_items_.clear();
  
  std::map <std::string, AxesItem>::iterator axes_iter;
  for ( axes_iter = axes_items_.begin(); axes_iter != axes_items_.end(); axes_iter++ )
  {
    if ( axes_iter->second.list != GL_INVALID_VALUE )
      glDeleteLists ( axes_iter->second.list, 1 );
    for ( int i = 0; i < 7; i++ )
      gluDeleteQuadric ( axes_iter->second.qobjs[i] );
  }
  
  axes_items_.clear();
}

void OpenGL3DScene::updateAllStructure()
{
  QMutexLocker locker ( &mutex_ );
  std::vector< std::string > structure_names;
  std::map <std::string, PathItem>::iterator path_iter;
  for ( path_iter = path_items_.begin(); path_iter != path_items_.end(); path_iter++ )
    structure_names.push_back(path_iter->first);

  std::map <std::string, LineItem>::iterator lines_iter;
  for ( lines_iter = line_items_.begin(); lines_iter != line_items_.end(); lines_iter++ )
    structure_names.push_back(lines_iter->first);
  
  std::map <std::string, CloudItem>::iterator cloud_iter;
  for ( cloud_iter = cloud_items_.begin(); cloud_iter != cloud_items_.end(); cloud_iter++ )
    structure_names.push_back(cloud_iter->first);
  
  std::map <std::string, AxesItem>::iterator axes_iter;
  for ( axes_iter = axes_items_.begin(); axes_iter != axes_items_.end(); axes_iter++ )
    structure_names.push_back(axes_iter->first);

  for( int i = 0; i < structure_names.size(); i++)
    updateStructure ( structure_names[i], false );

  updateGL();
}

void OpenGL3DScene::keyPressEvent ( QKeyEvent * event )
{

  QWidget::keyPressEvent ( event );
  makeCurrent();
  
  switch ( event->key() )
  {
  case Qt::Key_Left:
    moveCameraLeft ( t_inc_ );
    break;
  case Qt::Key_Right:
    moveCameraRight ( t_inc_ );
    break;
  case Qt::Key_Up:
    if ( fp_view_ )
      moveCameraForward ( t_inc_ );
    else
      moveCameraUp ( t_inc_ );
    break;
  case Qt::Key_Down:
    if ( fp_view_ )
      moveCameraBackward ( t_inc_ );
    else
      moveCameraDown ( t_inc_ );
    break;
  case Qt::Key_PageUp :
    if ( fp_view_ )
      moveCameraUp ( t_inc_ );
    else
      moveCameraForward ( t_inc_ );
    break;
  case Qt::Key_PageDown :
    if ( fp_view_ )
      moveCameraDown ( t_inc_ );
    else
      moveCameraBackward ( t_inc_ );
    break;
  case Qt::Key_A:
    if ( fp_view_ )
      turnCameraCCWY ( r_inc_ );
    break;
  case Qt::Key_S:
    if ( fp_view_ )
      turnCameraCWY ( r_inc_ );
    break;
  case Qt::Key_Q:
    if ( !fp_view_ )
      turnCameraCWZ( r_inc_ );
    break;
  case Qt::Key_W:
    if ( !fp_view_ )
      turnCameraCCWZ( r_inc_ );
    break;
  }
}

void OpenGL3DScene::wheelEvent ( QWheelEvent * event )
{
  QWidget::wheelEvent ( event );
  int num_degrees = event->delta() / 8;
  int num_steps = num_degrees / 15;

  ( num_steps > 0 ) ?moveCameraForward ( num_steps*t_inc_ ) :moveCameraBackward ( abs ( num_steps ) *t_inc_ );
  event->accept();
}

void OpenGL3DScene::mousePressEvent ( QMouseEvent* event )
{
  QWidget::mousePressEvent ( event );
  if( event->button() == Qt::LeftButton )
    left_mouse_button_pressed_ = true;
  else if( event->button() == Qt::RightButton )
    right_mouse_button_pressed_ = true;
  
  mouse_init_x_ =  event->x();
  mouse_init_y_ =  event->y();
}

void OpenGL3DScene::mouseReleaseEvent ( QMouseEvent* event )
{
  QWidget::mouseReleaseEvent( event );
  left_mouse_button_pressed_ = false;
  right_mouse_button_pressed_ = false;
}

void OpenGL3DScene::mouseMoveEvent ( QMouseEvent* event )
{
  QWidget::mouseMoveEvent ( event );
  GLfloat mouse_inc_x = GLfloat(event->x() - mouse_init_x_);
  GLfloat mouse_inc_y = GLfloat(event->y() - mouse_init_y_);
  mouse_inc_x /= GLfloat(this->geometry().width());
  mouse_inc_y /= GLfloat(this->geometry().height());
  
  if( left_mouse_button_pressed_ )
  {
    moveCameraLeft( 2*mouse_inc_x );
    moveCameraUp( 2*mouse_inc_y );
    mouse_init_x_ = event->x();
    mouse_init_y_ = event->y();
  }
  else if( right_mouse_button_pressed_ )
  {
    if ( fp_view_ )
      turnCameraCCWY( 100*mouse_inc_x );
    else
      turnCameraCCWZ( 100*mouse_inc_x );
    //moveCameraUp( 2*mouse_inc_y );
    mouse_init_x_ = event->x();
    mouse_init_y_ = event->y();    
  }
}
