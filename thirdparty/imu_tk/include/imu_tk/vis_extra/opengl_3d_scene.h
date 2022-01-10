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
#include <vector>
#include <map>

#include <QGLWidget>
#include <QVector>
#include <QMutex>
#include <Eigen/Geometry>

#include <limits>
#include <GL/glu.h>
#include "gl_camera.h"

/** @brief This class provides to the visualization inside a QWidget of simple 3D structures,
           using the OpenGL library */
class OpenGL3DScene: public QGLWidget
{
  
  Q_OBJECT
  
  public:
        
    OpenGL3DScene( QWidget* parent = 0, QColor background_color = QColor(0,0,0), 
                   bool global_axes = true, QColor global_frame_color = QColor(255,255,255));
    
    ~OpenGL3DScene();
    
    void registerPath( std::string name, QColor path_color = QColor(255,255,255) );
    void unregisterPath( std::string name );
    void pushPosition( std::string name, const Eigen::Vector3d &t );

    void registerLine( std::string name, QColor line_color = QColor(255,255,255) );
    void unregisterLine( std::string name );
    void setLine( std::string name, const Eigen::Vector3d& p0, const Eigen::Vector3d& p1 );
    
    void registerCloud( std::string name, QColor cloud_color = QColor(255,255,255) );
    void unregisterCloud( std::string name );
    void pushCloud( std::string name, const std::vector< Eigen::Vector3d > &pts );
    
    void registerAxes( std::string name, QColor axes_color = QColor(255,255,255) );
    void unregisterAxes( std::string name );
    void setAxesPos( std::string name, const Eigen::Vector3d &r, const Eigen::Vector3d &t );

    void unregisterAllStructures();
    
    void resetScene() 
    {
      //TODO
      max_x_ = max_y_= -std::numeric_limits< float >::max();
      min_x_ = min_y_ =std::numeric_limits< float >::min();
    };
    
    void updateNow();
    
    void enableGlobalAxes( bool enable, QColor color = QColor(0,0,0) );
    void setCameraIncrements(float translation_inc, float rotation_inc )
    {
      t_inc_ = GLfloat(translation_inc);
      r_inc_ = GLfloat(rotation_inc);
    };
    
    
  protected:
    
    /*! 
    @brief Sets up the OpenGL rendering context, defines display lists, etc. (QGLWidget).
    
    Gets called once before the first time resizeGL() or paintGL() is called.
    DON'T call this function directly!
     */
    void initializeGL();
    /*! 
    @brief Renders the OpenGL scene (QGLWidget).
    
    Gets called whenever the widget needs to be updated.
    DON'T call this function directly! If you need to trigger a repaint, you should call the widget's updateGL() function.
     */
    void paintGL();
    
    /*! 
    @brief Sets up the OpenGL viewport, projection, etc. (QGLWidget).
    
    @param width  New width of the widget
    @param height New height of the widget
    
    Gets called whenever the widget has been resized (and also when it is shown for the first time because all newly 
    created widgets get a resize event automatically).
    DON'T call this function directly!
     */
    void resizeGL(int width, int height);
    
    /*!
    @brief Event handler to receive key press events for the widget.
    
    @param event The key press event.
    */
    virtual void keyPressEvent ( QKeyEvent * event );
    virtual void wheelEvent ( QWheelEvent * event );
    virtual void mousePressEvent ( QMouseEvent * event );
    virtual void mouseReleaseEvent ( QMouseEvent * event );
    virtual void mouseMoveEvent ( QMouseEvent * event );
  private:
    /*! 
    @brief Compute the translation of the GL "camera" that should be performed in order to view all the 
           scene (only for global upper view, see setGlobalView() ).
    */
    void autoAdjustGView();

    
    //! Mutex varable for data synchronization.
    QMutex mutex_;
    //! Flag used to activate automatic adjustment that allows to view all the scene (only for global upper view, see setGlobalView() ).
    bool auto_adjust_;
    float auto_adjust_offset_;
    //! The utility object used to move inside an Open GL scene.
    GLCamera camera_;
    //! The background color of the scene
    QColor bg_color_;
    //! Draw x,y,z-axis flag
    bool draw_axes_;
    //! First person view flag
    bool fp_view_ ;
    
    struct PathItem
    {
      PathItem(){ removed = false; };
      std::vector< Eigen::Vector3d > unprocessed_poses;
      std::vector< GLuint > lists;
      QColor color;
      bool removed; 
    };
    
    struct LineItem
    {
      LineItem(){ updated = removed = false; };
      Eigen::Vector3d p0, p1;
      GLuint list;
      QColor color;
      bool updated, removed; 
    };
    
    struct AxesItem
    {
      AxesItem(){ updated = removed = false; };
      Eigen::Vector3d r, t;
      QColor color;
      bool updated;
      GLuint list;
      GLUquadric *qobjs[7];
      bool removed;
    };

    struct CloudItem
    {
      CloudItem(){ removed = false; };
      std::vector< Eigen::Vector3d > unprcessed_cloud;
      std::vector< GLuint > lists;
      QColor color;
      bool removed;
    };
    
    std::map <std::string, PathItem> path_items_;
    std::map <std::string, LineItem> line_items_;
    std::map <std::string, CloudItem> cloud_items_;
    std::map <std::string, AxesItem> axes_items_;
    
    //! The filed of view angle of the GL "camera".
    GLfloat view_angle_;
    //! The current form factor of the view port.
    GLfloat form_factor_;
    //! 2D boundaries (X,Y) of the current drawn scene.
    GLfloat max_x_, max_y_, min_x_, min_y_;
    //! x,y,z-axes OpenGL display list
    GLuint axis_list_;
    //! Translation increment while moving the camera
    GLfloat t_inc_;
    //! Rotation increment while moving the camera
    GLfloat r_inc_;

    bool left_mouse_button_pressed_, right_mouse_button_pressed_;
    int mouse_init_x_, mouse_init_y_;
    
signals:
    
    void zoomChanged(int value);
    void sceneUpdated();

  public slots:
    
    void updateStructure( std::string name, bool update_gl = true );
    void updateAllStructure();
    
    /*! 
    @brief Move and rotate the camera.
    
    @param tx Camera X position
    @param ty Camera Y position
    @param tz Camera Z position
    @param r1x First rotation (degrees) around X axis
    @param r2y Second rotation (degrees) around Y axis
    @param r3z Third rotation (degrees) around Z axis
    
    Camera is moved to the (tx, ty, tz) position then: 1) it is rotated of r1x degrees arount X axis;
    2) it is rotated of r2y degrees arount Y axis and 3) finally it is rotated of r3z degrees arount Z axis
    */
    void moveCamera(GLfloat tx, GLfloat ty, GLfloat tz, GLfloat r1x, GLfloat r2y, GLfloat r3z); 

    /*! 
    @brief Enable or disable the automatic adjustment of the viewport in order to show the whole scene 
           (only for global upper view, see setGlobalView()).
     */
    void setAutoAdjust(bool enable, float offset = 20.0 ) { auto_adjust_ = enable; auto_adjust_offset_ = offset; };
    
    
    /*! 
    @brief Set the global upper view.
     */
    void setGlobalView();
    
    /*! 
    @brief Set the first person view.
    
    @param good_pos Move the camera in a convenient position for the first person view
     */
    void setFirstPersonView( bool good_pos = true );
        
    void setZoom( int z );
    
    void moveCameraUp( GLfloat inc = 0.6f ) 
    { QMutexLocker locker(&mutex_); camera_.moveUpward(inc); updateGL(); };
    void moveCameraDown( GLfloat inc = 0.6f ) 
    { QMutexLocker locker(&mutex_); camera_.moveUpward(-inc); updateGL(); };
    void moveCameraLeft( GLfloat inc = 0.6f ) 
    { QMutexLocker locker(&mutex_);camera_.strafeRight(-inc); updateGL(); };
    void moveCameraRight( GLfloat inc = 0.6f ) 
    { QMutexLocker locker(&mutex_); camera_.strafeRight(inc); updateGL(); };
    void moveCameraForward( GLfloat inc = 0.6f ) 
    { QMutexLocker locker(&mutex_); camera_.moveForward( inc ); updateGL(); };
    void moveCameraBackward(GLfloat inc = 0.6f) 
    { QMutexLocker locker(&mutex_); camera_.moveForward( -inc ); updateGL(); };
    void turnCameraCCWY(GLfloat inc = 5.0f) 
    { QMutexLocker locker(&mutex_); camera_.rotateY(inc); updateGL(); };
    void turnCameraCWY(GLfloat inc = 5.0f) 
    { QMutexLocker locker(&mutex_); camera_.rotateY(-inc); updateGL(); };
    void turnCameraCCWX(GLfloat inc = 5.0f) 
    { QMutexLocker locker(&mutex_); camera_.rotateX(inc); updateGL(); };
    void turnCameraCWX(GLfloat inc = 5.0f) 
    { QMutexLocker locker(&mutex_); camera_.rotateX(-inc); updateGL(); };
    void turnCameraCCWZ(GLfloat inc = 5.0f) 
    { QMutexLocker locker(&mutex_); camera_.rotateZ(inc); updateGL(); };
    void turnCameraCWZ(GLfloat inc = 5.0f) 
    { QMutexLocker locker(&mutex_); camera_.rotateZ(-inc); updateGL(); };

};
