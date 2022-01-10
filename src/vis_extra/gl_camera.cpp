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

#include "imu_tk/vis_extra/gl_camera.h"

#include <math.h>
#include <GL/glu.h>

GLCamera::GLCamera()
{
  reset();
}

void GLCamera::reset()
{
  position_ = GLCameraUtils::vector3D ( 0.0f, 0.0f, 0.0f );
  view_dir_ = GLCameraUtils::vector3D ( 0.0f, 0.0f, -1.0f );
  right_vector_ = GLCameraUtils::vector3D ( 1.0f, 0.0f, 0.0f );
  up_vector_ = GLCameraUtils::vector3D ( 0.0f, 1.0f, 0.0f );

  rot_x = rot_y = rot_z = 0.0f;
}

// std::string GLCamera :: toString()
// {
//   std::stringstream ss;
//   ss<<"Custom Camera"<<std::endl;
//   ss<<"position: "<<position.toString()<<std::endl;
//   ss<<"view_dir: "<<view_dir.toString()<<std::endl;
//   ss<<"up_vector: "<<up_vector.toString()<<std::endl;
//   ss<<"right_vector: "<<right_vector.toString()<<std::endl;
//   ss<<"rotate: "<<rot_x<<" , "<<rot_y<<" , "<<rot_z<<std::endl;
//   return ss.str();
// }

void GLCamera::move ( GLfloat tx, GLfloat ty, GLfloat tz )
{
  GLCameraUtils::Vector3D T = GLCameraUtils::vector3D ( tx, ty, tz );
  position_ = position_ + T;
}

void GLCamera::moveAbs ( GLfloat tx, GLfloat ty, GLfloat tz )
{
  GLCameraUtils::Vector3D T = GLCameraUtils::vector3D ( tx, ty, tz );
  position_ = T;
}

void GLCamera::rotateX ( GLfloat angle )
{
  rot_x += angle;

  view_dir_ = GLCameraUtils::normalizeVector3D ( view_dir_*cos ( angle*PIdiv180 )
             + up_vector_*sin ( angle*PIdiv180 ) );

  up_vector_ = GLCameraUtils::crossProduct ( &view_dir_, &right_vector_ ) *-1;


}

void GLCamera::rotateY ( GLfloat angle )
{
  rot_y += angle;

  view_dir_ = GLCameraUtils::normalizeVector3D ( view_dir_*cos ( angle*PIdiv180 )
             - right_vector_*sin ( angle*PIdiv180 ) );

  right_vector_ = GLCameraUtils::crossProduct ( &view_dir_, &up_vector_ );
}

void GLCamera::rotateZ ( GLfloat angle )
{
  rot_z += angle;

  right_vector_ = GLCameraUtils::normalizeVector3D ( right_vector_*cos ( angle*PIdiv180 )
                 + up_vector_*sin ( angle*PIdiv180 ) );

  up_vector_ = GLCameraUtils::crossProduct ( &view_dir_, &right_vector_ ) *-1;
}

void GLCamera::render()
{
  GLCameraUtils::Vector3D view_point = position_+view_dir_;

  gluLookAt ( position_.x,position_.y,position_.z,
              view_point.x,view_point.y,view_point.z,
              up_vector_.x,up_vector_.y,up_vector_.z );

}

void GLCamera::moveForward ( GLfloat distance )
{
  position_ = position_ + ( view_dir_*distance );
}

void GLCamera::strafeRight ( GLfloat distance )
{
  position_ = position_ + ( right_vector_*distance );
}

void GLCamera::moveUpward ( GLfloat distance )
{
  position_ = position_ + ( up_vector_*distance );
}

GLCameraUtils::Vector3D GLCameraUtils::vector3D ( GLfloat x, GLfloat y, GLfloat z )
{
  Vector3D tmp;
  tmp.x = x;
  tmp.y = y;
  tmp.z = z;
  return tmp;
}

GLfloat GLCameraUtils::getVector3DLength ( Vector3D * v )
{
  return ( GLfloat ) ( sqrt ( ( v->x ) * ( v->x ) + ( v->y ) * ( v->y ) + ( v->z ) * ( v->y ) ) );
}

GLCameraUtils::Vector3D GLCameraUtils::normalizeVector3D ( Vector3D v )
{
  Vector3D res;
  float l = GLCameraUtils::getVector3DLength ( &v );
  if ( l == 0.0f ) return vector3D ( 0.0f,0.0f,0.0f );
  res.x = v.x / l;
  res.y = v.y / l;
  res.z = v.z / l;
  return res;
}

GLCameraUtils::Vector3D GLCameraUtils::operator + ( Vector3D v, Vector3D u )
{
  Vector3D res;
  res.x = v.x+u.x;
  res.y = v.y+u.y;
  res.z = v.z+u.z;
  return res;
}

GLCameraUtils::Vector3D GLCameraUtils::operator - ( Vector3D v, Vector3D u )
{
  Vector3D res;
  res.x = v.x-u.x;
  res.y = v.y-u.y;
  res.z = v.z-u.z;
  return res;
}

GLCameraUtils::Vector3D GLCameraUtils::operator * ( Vector3D v, float r )
{
  Vector3D res;
  res.x = v.x*r;
  res.y = v.y*r;
  res.z = v.z*r;
  return res;
}

GLCameraUtils::Vector3D GLCameraUtils::crossProduct ( Vector3D * u, Vector3D * v )
{
  Vector3D res;
  res.x = u->y*v->z - u->z*v->y;
  res.y = u->z*v->x - u->x*v->z;
  res.z = u->x*v->y - u->y*v->x;

  return res;
}

float GLCameraUtils::operator * ( Vector3D v, Vector3D u )
{
  return v.x*u.x+v.y*u.y+v.z*u.z;
}



