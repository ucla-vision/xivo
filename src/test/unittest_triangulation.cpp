#include <gtest/gtest.h>
#include "helpers.h"

using namespace Eigen;
using namespace xivo;

#include "core.h"
#include "math.h"

class Triangulation : public :: testing :: Test
{
  protected:
    float max_theta_thresh = 0.1 * M_PI / 180;
    float beta_thresh = 0.25 * M_PI / 180;
    float eps = 0.5;
};

TEST_F(Triangulation, Normal_Inputs) {
  /*
  3D point = (2,3,5)

  Translation = 10

  Rotation = 10 degree around y axis
  */

  Vec2 xc1{0.4, 0.6};
  float z1 = 5;

  Vec4 Xc1_homo{xc1[0] * z1, xc1[1] * z1, z1, 1};

  Mat4 g21;

  g21 << 0.9849082,0,0.1731,-9.8490,
        0,1,0,0,
        -0.17310,0,0.98490,1.73101,
        0,0,0,1;

  Vec4 Xc2_homo = g21 * Xc1_homo;

  Vec2 xc2{Xc2_homo[0]/Xc2_homo[2], Xc2_homo[1]/Xc2_homo[2]};

  Mat4 pose_homo = g21.inverse();
  SE3 g12 = SE3::fitToSE3(pose_homo);

  Vec3 Xc1;

  bool return_output = L1Angular(g12, xc1, xc2, Xc1, max_theta_thresh, beta_thresh);
  EXPECT_TRUE(return_output);
  EXPECT_LE(abs(Xc1[2] - z1), eps);


  return_output = L2Angular(g12, xc1, xc2, Xc1, max_theta_thresh, beta_thresh);
  EXPECT_TRUE(return_output);
  EXPECT_LE(abs(Xc1[2] - z1), eps);

  return_output = LinfAngular(g12, xc1, xc2, Xc1, max_theta_thresh, beta_thresh);
  EXPECT_TRUE(return_output);
  EXPECT_LE(abs(Xc1[2] - z1), eps);
}



TEST_F(Triangulation, Parallax) {

  Vec2 xc1{2.2, 0.7};
  float z1 = 5;

  Vec4 Xc1_homo{xc1[0] * z1, xc1[1] * z1, z1, 1};

  Mat4 g21;

  g21 << 0.9998,0,0.01745,-0.01,
        0,1,0,0,
        -0.01745,0,0.9998,0,
        0,0,0,1;

  Vec4 Xc2_homo = g21 * Xc1_homo;

  Vec2 xc2{Xc2_homo[0]/Xc2_homo[2], Xc2_homo[1]/Xc2_homo[2]};

  Mat4 pose_homo = g21.inverse();
  SE3 g12 = SE3::fitToSE3(pose_homo);

  Vec3 Xc1;
  bool return_output = L1Angular(g12, xc1, xc2, Xc1, max_theta_thresh, beta_thresh);
  EXPECT_FALSE(return_output);

  return_output = L2Angular(g12, xc1, xc2, Xc1, max_theta_thresh, beta_thresh);
  EXPECT_FALSE(return_output);

  return_output = LinfAngular(g12, xc1, xc2, Xc1, max_theta_thresh, beta_thresh);
  EXPECT_FALSE(return_output);

}


TEST_F(Triangulation, Cheirality) {

  Vec2 xc1{2, -0.77};
  float z1 = 5;

  Vec4 Xc1_homo{xc1[0] * z1, xc1[1] * z1, z1, 1};

  Mat4 g21;

  g21 << -1,0,0,3,
        0,1,0,0,
        0,0,-1,0,
        0,0,0,1;

  Vec4 Xc2_homo = g21 * Xc1_homo;

  Vec2 xc2{Xc2_homo[0]/Xc2_homo[2], Xc2_homo[1]/Xc2_homo[2]};

  Mat4 pose_homo = g21.inverse();
  SE3 g12 = SE3::fitToSE3(pose_homo);

  Vec3 Xc1;

  bool return_output = L1Angular(g12, xc1, xc2, Xc1, max_theta_thresh, beta_thresh);
  EXPECT_FALSE(return_output);

  return_output = L2Angular(g12, xc1, xc2, Xc1, max_theta_thresh, beta_thresh);
  EXPECT_FALSE(return_output);

  return_output = LinfAngular(g12, xc1, xc2, Xc1, max_theta_thresh, beta_thresh);
  EXPECT_FALSE(return_output);

}

TEST_F(Triangulation, Angular_Reprojection_Error) {

  Vec2 xc1{2.22216, 0.778023};
  float z1 = 5;

  Vec4 Xc1_homo{xc1[0] * z1, xc1[1] * z1, z1, 1};

  Mat4 g21;

  g21 << 1,0,0,3,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1;

  Vec4 Xc2_homo = g21 * Xc1_homo;

  float noise = 0.7;

  Vec2 xc2{Xc2_homo[0]/Xc2_homo[2] + noise, Xc2_homo[1]/Xc2_homo[2] + noise};

  Mat4 pose_homo = g21.inverse();
  SE3 g12 = SE3::fitToSE3(pose_homo);

  Vec3 Xc1;

  // NOTE (April 2022): On Arch Linux (gcc11) and Ubuntu 20.04 (gcc9), this test
  // (for L1Angular) fails when compiled in RELEASE mode but passes when
  // compiled in DEBUG mode.
  bool return_output = L1Angular(g12, xc1, xc2, Xc1, max_theta_thresh, beta_thresh);
  EXPECT_FALSE(return_output);

  return_output = L2Angular(g12, xc1, xc2, Xc1, max_theta_thresh, beta_thresh);
  EXPECT_FALSE(return_output);

  return_output = LinfAngular(g12, xc1, xc2, Xc1, max_theta_thresh, beta_thresh);
  EXPECT_FALSE(return_output);

}



TEST_F(Triangulation, Vanishing_Point) {

  Vec2 xc1{0.2, 0.3};
  float z1 = 6000;

  Vec4 Xc1_homo{xc1[0] * z1, xc1[1] * z1, z1, 1};

  Mat4 g21;

  g21 << 1,0,0,-0.1,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1;

  Vec4 Xc2_homo = g21 * Xc1_homo;

  float z2 = Xc2_homo[2];

  Vec2 xc2{Xc2_homo[0]/Xc2_homo[2], Xc2_homo[1]/Xc2_homo[2]};

  Mat4 pose_homo = g21.inverse();
  SE3 g12 = SE3::fitToSE3(pose_homo);

  Vec3 Xc1;

  bool return_output = L1Angular(g12, xc1, xc2, Xc1, max_theta_thresh, beta_thresh);
  EXPECT_FALSE(return_output);

  return_output = L2Angular(g12, xc1, xc2, Xc1, max_theta_thresh, beta_thresh);
  EXPECT_FALSE(return_output);

  return_output = LinfAngular(g12, xc1, xc2, Xc1, max_theta_thresh, beta_thresh);
  EXPECT_FALSE(return_output);

}