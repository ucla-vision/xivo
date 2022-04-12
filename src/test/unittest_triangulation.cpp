#include <gtest/gtest.h>
#include "helpers.h"

using namespace Eigen;
using namespace xivo;

#include "core.h"
#include "math.h"

TEST(Triangulation, Normal_Inputs) {

  /*
  3D point = (2,3,5)

  Translation = 10

  Rotation = 10 degree around y axis
  */

  Vec2 xc1{0.4, 0.6};
  int depth = 5;

  Vec4 xc1_homo{xc1[0] * depth, xc1[1] * depth, depth, 1};

  Mat4 g21;

  g21 << 0.9849082,0,0.1731,-9.8490,
        0,1,0,0,
        -0.17310,0,0.98490,1.73101,
        0,0,0,1;

  Vec4 xc2_homo = g21 * xc1_homo;

  int new_depth = xc2_homo[2];

  Vec2 xc2{xc2_homo[0]/xc2_homo[2], xc2_homo[1]/xc2_homo[2]};

  Mat4 pose_homo;

  pose_homo = g21.inverse();

  Mat34 pose;

  pose << pose_homo.coeff(0,0),pose_homo.coeff(0,1),pose_homo.coeff(0,2),pose_homo.coeff(0,3),
          pose_homo.coeff(1,0),pose_homo.coeff(1,1),pose_homo.coeff(1,2),pose_homo.coeff(1,3),
          pose_homo.coeff(2,0),pose_homo.coeff(2,1),pose_homo.coeff(2,2),pose_homo.coeff(2,3);

  SE3 g12{pose};

  Vec3 Xc1;

  bool return_output = L1Angular(g12, xc1, xc2, Xc1);

  EXPECT_TRUE(return_output);
  EXPECT_LE((Xc1[2] - new_depth), 1);

}



TEST(Triangulation, Parallax) {

  Vec2 xc1{2.2, 0.7};
  int depth = 5;

  Vec4 xc1_homo{xc1[0] * depth, xc1[1] * depth, depth, 1};

  Mat4 g21;

  g21 << 0.999444,0,-0.03399,-0.99944,
        0,1,0,0,
        	0.033991,0,0.999444,	-0.033991,
        0,0,0,1;

  Vec4 xc2_homo = g21 * xc1_homo;

  int new_depth = xc2_homo[2];

  Vec2 xc2{xc2_homo[0]/xc2_homo[2], xc2_homo[1]/xc2_homo[2]};

  Mat4 pose_homo;

  pose_homo = g21.inverse();

  Mat34 pose;

  pose << pose_homo.coeff(0,0),pose_homo.coeff(0,1),pose_homo.coeff(0,2),pose_homo.coeff(0,3),
          pose_homo.coeff(1,0),pose_homo.coeff(1,1),pose_homo.coeff(1,2),pose_homo.coeff(1,3),
          pose_homo.coeff(2,0),pose_homo.coeff(2,1),pose_homo.coeff(2,2),pose_homo.coeff(2,3);

  SE3 g12{pose};

  Vec3 Xc1;
  bool return_output = L1Angular(g12, xc1, xc2, Xc1);

  EXPECT_FALSE(return_output);

}


TEST(Triangulation, Cheirality) {

  Vec2 xc1{2, -0.77};
  int depth = 5;

  Vec4 xc1_homo{xc1[0] * depth, xc1[1] * depth, depth, 1};

  Mat4 g21;

  g21 << -1,0,0,3,
        0,1,0,0,
        0,0,-1,0,
        0,0,0,1;

  Vec4 xc2_homo = g21 * xc1_homo;

  int new_depth = xc2_homo[2];

  Vec2 xc2{xc2_homo[0]/xc2_homo[2], xc2_homo[1]/xc2_homo[2]};

  Mat4 pose_homo;

  pose_homo = g21.inverse();

  Mat34 pose;

  pose << pose_homo.coeff(0,0),pose_homo.coeff(0,1),pose_homo.coeff(0,2),pose_homo.coeff(0,3),
          pose_homo.coeff(1,0),pose_homo.coeff(1,1),pose_homo.coeff(1,2),pose_homo.coeff(1,3),
          pose_homo.coeff(2,0),pose_homo.coeff(2,1),pose_homo.coeff(2,2),pose_homo.coeff(2,3);

  SE3 g12{pose};

  Vec3 Xc1;

  bool return_output = L1Angular(g12, xc1, xc2, Xc1);

  EXPECT_FALSE(return_output);

}

TEST(Triangulation, Angular_Reprojection_Error) {

  Vec2 xc1{2.22216, 0.778023};
  int depth = 5;

  Vec4 xc1_homo{xc1[0] * depth, xc1[1] * depth, depth, 1};

  Mat4 g21;

  g21 << 1,0,0,3,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1;

  Vec4 xc2_homo = g21 * xc1_homo;

  int new_depth = xc2_homo[2];
  float noise = 0.7;

  Vec2 xc2{xc2_homo[0]/xc2_homo[2] + noise, xc2_homo[1]/xc2_homo[2] + noise};

  Mat4 pose_homo;

  pose_homo = g21.inverse();

  Mat34 pose;

  pose << pose_homo.coeff(0,0),pose_homo.coeff(0,1),pose_homo.coeff(0,2),pose_homo.coeff(0,3),
          pose_homo.coeff(1,0),pose_homo.coeff(1,1),pose_homo.coeff(1,2),pose_homo.coeff(1,3),
          pose_homo.coeff(2,0),pose_homo.coeff(2,1),pose_homo.coeff(2,2),pose_homo.coeff(2,3);

  SE3 g12{pose};

  Vec3 Xc1;

  bool return_output = L1Angular(g12, xc1, xc2, Xc1);

  EXPECT_FALSE(return_output);

}



TEST(Triangulation, Vanishing_Point) {

  Vec2 xc1{2,3};
  Vec2 xc2{2,3};

  Mat34 pose;
  pose << 1,0,0,-3,
          0,1,0,0,
          0,0,1,0;

  SE3 g12{pose};

  Vec3 Xc1;

  bool return_output = L1Angular(g12, xc1, xc2, Xc1);

  EXPECT_FALSE(return_output);
  EXPECT_TRUE(isnan(Xc1[2]));

}