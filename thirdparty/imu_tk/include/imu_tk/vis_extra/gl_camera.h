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

#include <cmath>
#include <GL/gl.h>

#define PIdiv180 GLfloat(3.1415265359)/GLfloat(180.0)


/*! @brief Utility structures and functions used by GLCamera object */
namespace GLCameraUtils
{
/*! @brief Basic 3D vector */
struct Vector3D
{
  GLfloat x,y,z;
};

/*! @brief Return a Vector3D with the given coordinates */
inline Vector3D vector3D ( GLfloat x, GLfloat y, GLfloat z );

/*! @brief Compute the length of a Vector3D. */
inline GLfloat getVector3DLength ( Vector3D * v );

/*! @brief Normalize the coordinates of a Vector3D to the unit vector. */
inline Vector3D normalizeVector3D ( Vector3D v );

/*! @brief Computes the sum between two Vector3D */
inline Vector3D operator+ ( Vector3D v, Vector3D u );

/*! @brief Computes the difference between two Vector3D */
inline Vector3D operator- ( Vector3D v, Vector3D u );

/*! @brief Computes the scalar multiplication between a Vector3D and a scalar */
inline Vector3D operator* ( Vector3D v, float r );

/*! @brief Computes the cross product between two Vector3D */
inline Vector3D crossProduct ( Vector3D * u, Vector3D * v );

/*! @brief Computes the dot product between two Vector3D */
inline float operator* ( Vector3D v, Vector3D u );
};

/*!
@brief This is an utility class to move inside an Open GL scene.
 */
class GLCamera
{
public:
  /*! @brief Object constructor. */
  GLCamera();
  /*! @brief Object destructor. */
  ~GLCamera() {};

  /*! @brief Reset the position of the camera to the frame origin. */
  void reset();

  /*! @brief Perform the camera movement, i.e. call the OpenGL functions to change the view */
  void render ();

  /*!
  @brief Move the camera relatively to the current position
  @param tx X coordinate of the translation vector
  @param ty y coordinate of the  translation vector
  @param tz z coordinate of the  translation vector
  */
  void move ( GLfloat tx, GLfloat ty, GLfloat tz );

  /*!
  @brief Translate the position relatively to the world frame
  @param tx X coordinate of the translation vector
  @param ty y coordinate of the  translation vector
  @param tz z coordinate of the  translation vector
  */
  void moveAbs ( GLfloat tx, GLfloat ty, GLfloat tz );

  /*!
  @brief Rotate around the X axis
  @param angle The angle in deg
  */
  void rotateX ( GLfloat angle );
  /*!
  @brief Rotate around the Y axis
  @param angle The angle in deg
   */
  void rotateY ( GLfloat angle );
  /*!
  @brief Rotate around the Z axis
  @param angle The angle in deg
   */
  void rotateZ ( GLfloat angle );

  /*!
  @brief Translate forward
  @param distance The distance to translate
   */
  void moveForward ( GLfloat distance );
  /*!
  @brief Translate upward
  @param distance The distance to translate
   */
  void moveUpward ( GLfloat distance );
  /*!
  @brief Translate sideways
  @param distance The distance to translate
  */
  void strafeRight ( GLfloat distance );

private:

  GLCameraUtils::Vector3D view_dir_;
  GLCameraUtils::Vector3D right_vector_;
  GLCameraUtils::Vector3D up_vector_;
  //! Current translation of the camera
  GLCameraUtils::Vector3D position_;

  //! Current rotation (euler angles) of the camera
  GLfloat rot_x, rot_y, rot_z;

};
