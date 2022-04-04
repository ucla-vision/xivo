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

  Vec2 xc1{2/5,3/5};
  Vec2 xc2{-8.7447 / 3.5397,3 / 3.5397};

  Mat34 pose;

  pose << 0.9849,0,-0.17310,10,
          0,1,0,0,
          0.17310	,0,0.9849,0;

  SE3 g12{pose};

  Vec3 Xc1;
  
  Xc1 = L1Angular(g12, xc1, xc2);

  EXPECT_LE((Xc1[2] - 3.5397),1);
}



TEST(Triangulation, Parallax) {

  Vec2 xc1{2.2,0.7};
  Vec2 xc2{3,0.8};

  Mat34 pose;
  pose << 0.996,0,-0.087,-0.9961,
          0,1,0,0,
          0.087,0,0.9961,-0.087;

  SE3 g12{pose};

  Vec3 Xc1;
  
  Xc1 = L1Angular(g12, xc1, xc2);

  EXPECT_FLOAT_EQ(0, 0);
}


TEST(Triangulation, Cheirality) {

  Vec2 xc1{2,-0.77};
  Vec2 xc2{-5,-1.77};

  Mat34 pose;
  pose << -1,0,0,3,
          0,1,0,0,
          0,0,-1,0;

  SE3 g12{pose};

  Vec3 Xc1;
  
  Xc1 = L1Angular(g12, xc1, xc2);

  EXPECT_FLOAT_EQ(0, 0);
}

TEST(Triangulation, Angular_Reprojection_Error) {

Vec2 xc1{2.22216,0.778023};
  Vec2 xc2{5.22216,1.778023};

  Mat34 pose;
  pose << 0.707,0,-0.707,3,
             0,1,0,1,
             0.707,0,0.707,0;

  SE3 g12{pose};

  Vec3 Xc1;
  
  Xc1 = L1Angular(g12, xc1, xc2);

  EXPECT_FLOAT_EQ(0, 0);
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
  
  Xc1 = L1Angular(g12, xc1, xc2);

  EXPECT_TRUE(isnan(Xc1[2]));
}