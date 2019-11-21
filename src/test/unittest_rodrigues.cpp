#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>

#include "alias.h"
#include "rodrigues.h"
#include "timer.h"

using namespace Eigen;
using namespace xivo;


// Test Rodrigues and derivatives against OpenCV implementation for correctness
TEST(Rodrigues, Rodrigues) {

  // Test vectors
  Vec3 w0(0.5, 0.5, 0.7);
  Vec3 w1(0.1, 0.2, -0.1);
  Vec3 w2(0, 0, 0);

  // Output Jacobians
  Mat93 dR_dw0;
  Mat93 dR_dw1;
  Mat93 dR_dw2;

  // Our implementation
  Mat3 R0 = rodrigues(w0, &dR_dw0);
  Mat3 R1 = rodrigues(w1, &dR_dw1);
  Mat3 R2 = rodrigues(w2, &dR_dw2);

  // OpenCV implementation
  cv::Vec3d v0( w0(0), w0(1), w0(2) );
  cv::Matx33d RR0;
  cv::Mat_<number_t> dRR_dv0(3, 9, 0.0); // OpenCV does (cols, rows) to be like images
  cv::Rodrigues(v0, RR0, dRR_dv0);

  cv::Vec3d v1( w1(0), w1(1), w1(2) );
  cv::Matx33d RR1;
  cv::Mat_<number_t> dRR_dv1(3, 9, 0.0);
  cv::Rodrigues(v1, RR1, dRR_dv1);

  cv::Vec3d v2( w2(0), w2(1), w2(2) );
  cv::Matx33d RR2;
  cv::Mat_<number_t> dRR_dv2(3, 9, 0.0);
  cv::Rodrigues(v2, RR2, dRR_dv2);

  // Check for correctness
  EXPECT_FLOAT_EQ(R0(0,0), RR0(0,0));
  EXPECT_FLOAT_EQ(R0(0,1), RR0(0,1));
  EXPECT_FLOAT_EQ(R0(0,2), RR0(0,2));
  EXPECT_FLOAT_EQ(R0(1,0), RR0(1,0));
  EXPECT_FLOAT_EQ(R0(1,1), RR0(1,1));
  EXPECT_FLOAT_EQ(R0(1,2), RR0(1,2));
  EXPECT_FLOAT_EQ(R0(2,0), RR0(2,0));
  EXPECT_FLOAT_EQ(R0(2,1), RR0(2,1));
  EXPECT_FLOAT_EQ(R0(2,2), RR0(2,2));

  EXPECT_FLOAT_EQ(R1(0,0), RR1(0,0));
  EXPECT_FLOAT_EQ(R1(0,1), RR1(0,1));
  EXPECT_FLOAT_EQ(R1(0,2), RR1(0,2));
  EXPECT_FLOAT_EQ(R1(1,0), RR1(1,0));
  EXPECT_FLOAT_EQ(R1(1,1), RR1(1,1));
  EXPECT_FLOAT_EQ(R1(1,2), RR1(1,2));
  EXPECT_FLOAT_EQ(R1(2,0), RR1(2,0));
  EXPECT_FLOAT_EQ(R1(2,1), RR1(2,1));
  EXPECT_FLOAT_EQ(R1(2,2), RR1(2,2));

  EXPECT_FLOAT_EQ(R2(0,0), RR2(0,0));
  EXPECT_FLOAT_EQ(R2(0,1), RR2(0,1));
  EXPECT_FLOAT_EQ(R2(0,2), RR2(0,2));
  EXPECT_FLOAT_EQ(R2(1,0), RR2(1,0));
  EXPECT_FLOAT_EQ(R2(1,1), RR2(1,1));
  EXPECT_FLOAT_EQ(R2(1,2), RR2(1,2));
  EXPECT_FLOAT_EQ(R2(2,0), RR2(2,0));
  EXPECT_FLOAT_EQ(R2(2,1), RR2(2,1));
  EXPECT_FLOAT_EQ(R2(2,2), RR2(2,2));

  // Check Jacobians. Note that Eigen indexing is (row,col) while OpenCV is (col,row)
  EXPECT_FLOAT_EQ(dR_dw0(0,0), dRR_dv0.at<number_t>(0,0));
  EXPECT_FLOAT_EQ(dR_dw0(0,1), dRR_dv0.at<number_t>(1,0));
  EXPECT_FLOAT_EQ(dR_dw0(0,2), dRR_dv0.at<number_t>(2,0));
  EXPECT_FLOAT_EQ(dR_dw0(1,0), dRR_dv0.at<number_t>(0,1));
  EXPECT_FLOAT_EQ(dR_dw0(1,1), dRR_dv0.at<number_t>(1,1));
  EXPECT_FLOAT_EQ(dR_dw0(1,2), dRR_dv0.at<number_t>(2,1));
  EXPECT_FLOAT_EQ(dR_dw0(2,0), dRR_dv0.at<number_t>(0,2));
  EXPECT_FLOAT_EQ(dR_dw0(2,1), dRR_dv0.at<number_t>(1,2));
  EXPECT_FLOAT_EQ(dR_dw0(2,2), dRR_dv0.at<number_t>(2,2));
  EXPECT_FLOAT_EQ(dR_dw0(3,0), dRR_dv0.at<number_t>(0,3));
  EXPECT_FLOAT_EQ(dR_dw0(3,1), dRR_dv0.at<number_t>(1,3));
  EXPECT_FLOAT_EQ(dR_dw0(3,2), dRR_dv0.at<number_t>(2,3));
  EXPECT_FLOAT_EQ(dR_dw0(4,0), dRR_dv0.at<number_t>(0,4));
  EXPECT_FLOAT_EQ(dR_dw0(4,1), dRR_dv0.at<number_t>(1,4));
  EXPECT_FLOAT_EQ(dR_dw0(4,2), dRR_dv0.at<number_t>(2,4));
  EXPECT_FLOAT_EQ(dR_dw0(5,0), dRR_dv0.at<number_t>(0,5));
  EXPECT_FLOAT_EQ(dR_dw0(5,1), dRR_dv0.at<number_t>(1,5));
  EXPECT_FLOAT_EQ(dR_dw0(5,2), dRR_dv0.at<number_t>(2,5));
  EXPECT_FLOAT_EQ(dR_dw0(6,0), dRR_dv0.at<number_t>(0,6));
  EXPECT_FLOAT_EQ(dR_dw0(6,1), dRR_dv0.at<number_t>(1,6));
  EXPECT_FLOAT_EQ(dR_dw0(6,2), dRR_dv0.at<number_t>(2,6));
  EXPECT_FLOAT_EQ(dR_dw0(7,0), dRR_dv0.at<number_t>(0,7));
  EXPECT_FLOAT_EQ(dR_dw0(7,1), dRR_dv0.at<number_t>(1,7));
  EXPECT_FLOAT_EQ(dR_dw0(7,2), dRR_dv0.at<number_t>(2,7));
  EXPECT_FLOAT_EQ(dR_dw0(8,0), dRR_dv0.at<number_t>(0,8));
  EXPECT_FLOAT_EQ(dR_dw0(8,1), dRR_dv0.at<number_t>(1,8));
  EXPECT_FLOAT_EQ(dR_dw0(8,2), dRR_dv0.at<number_t>(2,8));

  EXPECT_FLOAT_EQ(dR_dw1(0,0), dRR_dv1.at<number_t>(0,0));
  EXPECT_FLOAT_EQ(dR_dw1(0,1), dRR_dv1.at<number_t>(1,0));
  EXPECT_FLOAT_EQ(dR_dw1(0,2), dRR_dv1.at<number_t>(2,0));
  EXPECT_FLOAT_EQ(dR_dw1(1,0), dRR_dv1.at<number_t>(0,1));
  EXPECT_FLOAT_EQ(dR_dw1(1,1), dRR_dv1.at<number_t>(1,1));
  EXPECT_FLOAT_EQ(dR_dw1(1,2), dRR_dv1.at<number_t>(2,1));
  EXPECT_FLOAT_EQ(dR_dw1(2,0), dRR_dv1.at<number_t>(0,2));
  EXPECT_FLOAT_EQ(dR_dw1(2,1), dRR_dv1.at<number_t>(1,2));
  EXPECT_FLOAT_EQ(dR_dw1(2,2), dRR_dv1.at<number_t>(2,2));
  EXPECT_FLOAT_EQ(dR_dw1(3,0), dRR_dv1.at<number_t>(0,3));
  EXPECT_FLOAT_EQ(dR_dw1(3,1), dRR_dv1.at<number_t>(1,3));
  EXPECT_FLOAT_EQ(dR_dw1(3,2), dRR_dv1.at<number_t>(2,3));
  EXPECT_FLOAT_EQ(dR_dw1(4,0), dRR_dv1.at<number_t>(0,4));
  EXPECT_FLOAT_EQ(dR_dw1(4,1), dRR_dv1.at<number_t>(1,4));
  EXPECT_FLOAT_EQ(dR_dw1(4,2), dRR_dv1.at<number_t>(2,4));
  EXPECT_FLOAT_EQ(dR_dw1(5,0), dRR_dv1.at<number_t>(0,5));
  EXPECT_FLOAT_EQ(dR_dw1(5,1), dRR_dv1.at<number_t>(1,5));
  EXPECT_FLOAT_EQ(dR_dw1(5,2), dRR_dv1.at<number_t>(2,5));
  EXPECT_FLOAT_EQ(dR_dw1(6,0), dRR_dv1.at<number_t>(0,6));
  EXPECT_FLOAT_EQ(dR_dw1(6,1), dRR_dv1.at<number_t>(1,6));
  EXPECT_FLOAT_EQ(dR_dw1(6,2), dRR_dv1.at<number_t>(2,6));
  EXPECT_FLOAT_EQ(dR_dw1(7,0), dRR_dv1.at<number_t>(0,7));
  EXPECT_FLOAT_EQ(dR_dw1(7,1), dRR_dv1.at<number_t>(1,7));
  EXPECT_FLOAT_EQ(dR_dw1(7,2), dRR_dv1.at<number_t>(2,7));
  EXPECT_FLOAT_EQ(dR_dw1(8,0), dRR_dv1.at<number_t>(0,8));
  EXPECT_FLOAT_EQ(dR_dw1(8,1), dRR_dv1.at<number_t>(1,8));
  EXPECT_FLOAT_EQ(dR_dw1(8,2), dRR_dv1.at<number_t>(2,8));

  EXPECT_FLOAT_EQ(dR_dw2(0,0), dRR_dv2.at<number_t>(0,0));
  EXPECT_FLOAT_EQ(dR_dw2(0,1), dRR_dv2.at<number_t>(1,0));
  EXPECT_FLOAT_EQ(dR_dw2(0,2), dRR_dv2.at<number_t>(2,0));
  EXPECT_FLOAT_EQ(dR_dw2(1,0), dRR_dv2.at<number_t>(0,1));
  EXPECT_FLOAT_EQ(dR_dw2(1,1), dRR_dv2.at<number_t>(1,1));
  EXPECT_FLOAT_EQ(dR_dw2(1,2), dRR_dv2.at<number_t>(2,1));
  EXPECT_FLOAT_EQ(dR_dw2(2,0), dRR_dv2.at<number_t>(0,2));
  EXPECT_FLOAT_EQ(dR_dw2(2,1), dRR_dv2.at<number_t>(1,2));
  EXPECT_FLOAT_EQ(dR_dw2(2,2), dRR_dv2.at<number_t>(2,2));
  EXPECT_FLOAT_EQ(dR_dw2(3,0), dRR_dv2.at<number_t>(0,3));
  EXPECT_FLOAT_EQ(dR_dw2(3,1), dRR_dv2.at<number_t>(1,3));
  EXPECT_FLOAT_EQ(dR_dw2(3,2), dRR_dv2.at<number_t>(2,3));
  EXPECT_FLOAT_EQ(dR_dw2(4,0), dRR_dv2.at<number_t>(0,4));
  EXPECT_FLOAT_EQ(dR_dw2(4,1), dRR_dv2.at<number_t>(1,4));
  EXPECT_FLOAT_EQ(dR_dw2(4,2), dRR_dv2.at<number_t>(2,4));
  EXPECT_FLOAT_EQ(dR_dw2(5,0), dRR_dv2.at<number_t>(0,5));
  EXPECT_FLOAT_EQ(dR_dw2(5,1), dRR_dv2.at<number_t>(1,5));
  EXPECT_FLOAT_EQ(dR_dw2(5,2), dRR_dv2.at<number_t>(2,5));
  EXPECT_FLOAT_EQ(dR_dw2(6,0), dRR_dv2.at<number_t>(0,6));
  EXPECT_FLOAT_EQ(dR_dw2(6,1), dRR_dv2.at<number_t>(1,6));
  EXPECT_FLOAT_EQ(dR_dw2(6,2), dRR_dv2.at<number_t>(2,6));
  EXPECT_FLOAT_EQ(dR_dw2(7,0), dRR_dv2.at<number_t>(0,7));
  EXPECT_FLOAT_EQ(dR_dw2(7,1), dRR_dv2.at<number_t>(1,7));
  EXPECT_FLOAT_EQ(dR_dw2(7,2), dRR_dv2.at<number_t>(2,7));
  EXPECT_FLOAT_EQ(dR_dw2(8,0), dRR_dv2.at<number_t>(0,8));
  EXPECT_FLOAT_EQ(dR_dw2(8,1), dRR_dv2.at<number_t>(1,8));
  EXPECT_FLOAT_EQ(dR_dw2(8,2), dRR_dv2.at<number_t>(2,8));

}



TEST(Rodrigues, InvRodrigues) {

  // Two test matrices - we're going to use the same cases as in the previous test,
  // except that we're going to test that we end up where we started. This is okay since
  // the code in rodrigues() and invrodrigues() do not overlap.
  Vec3 w0_orig(0.5, 0.5, 0.7);
  Vec3 w1_orig(0.1, 0.2, -0.1);
  Vec3 w2_orig(0.0, 0.0, 0.0);
  Mat3 R0 = rodrigues(w0_orig);
  Mat3 R1 = rodrigues(w1_orig);
  Mat3 R2 = rodrigues(w2_orig);

  // Derivatives
  Mat39 dw0_dR;
  Mat39 dw1_dR;
  Mat39 dw2_dR;

  // Result
  Vec3 w0 = invrodrigues(R0, &dw0_dR);
  Vec3 w1 = invrodrigues(R1, &dw1_dR);
  Vec3 w2 = invrodrigues(R2, &dw2_dR);

  EXPECT_FLOAT_EQ(w0(0), w0_orig(0));
  EXPECT_FLOAT_EQ(w0(1), w0_orig(1));
  EXPECT_FLOAT_EQ(w0(2), w0_orig(2));
  
  EXPECT_FLOAT_EQ(w1(0), w1_orig(0));
  EXPECT_FLOAT_EQ(w1(1), w1_orig(1));
  EXPECT_FLOAT_EQ(w1(2), w1_orig(2));

  EXPECT_FLOAT_EQ(w2(0), w2_orig(0));
  EXPECT_FLOAT_EQ(w2(1), w2_orig(1));
  EXPECT_FLOAT_EQ(w2(2), w2_orig(2));

  // Still going to use OpenCV to check that the Jacobians are implemented correctly
  cv::Vec3d v0;
  // I dunno if this initialization is row or column-major, but our matrices are
  // symmetric! :D
  cv::Matx33d RR0( R0(0,0), R0(0,1), R0(0,2),
                   R0(1,0), R0(1,1), R0(1,2),
                   R0(2,0), R0(2,1), R0(2,2));
  cv::Mat_<number_t> dv0_dRR(9, 3, 0.0); // OpenCV does (cols, rows) to be like images
  cv::Rodrigues(RR0, v0, dv0_dRR);

  cv::Vec3d v1;
  cv::Matx33d RR1( R1(0,0), R1(0,1), R1(0,2),
                   R1(1,0), R1(1,1), R1(1,2),
                   R1(2,0), R1(2,1), R1(2,2));
  cv::Mat_<number_t> dv1_dRR(9, 3, 0.0);
  cv::Rodrigues(RR1, v1, dv1_dRR);

  cv::Vec3d v2;
  cv::Matx33d RR2( R2(0,0), R2(0,1), R2(0,2),
                   R2(1,0), R2(1,1), R2(1,2),
                   R2(2,0), R2(2,1), R2(2,2));
  cv::Mat_<number_t> dv2_dRR(9, 3, 0.0);
  cv::Rodrigues(RR2, v2, dv2_dRR);

  // Sanity check to make sure I set up the OpenCV call correctly.
  EXPECT_FLOAT_EQ(v0(0), w0_orig(0));
  EXPECT_FLOAT_EQ(v0(1), w0_orig(1));
  EXPECT_FLOAT_EQ(v0(2), w0_orig(2));
  EXPECT_FLOAT_EQ(v1(0), w1_orig(0));
  EXPECT_FLOAT_EQ(v1(1), w1_orig(1));
  EXPECT_FLOAT_EQ(v1(2), w1_orig(2));
  EXPECT_FLOAT_EQ(v2(0), w2_orig(0));
  EXPECT_FLOAT_EQ(v2(1), w2_orig(1));
  EXPECT_FLOAT_EQ(v2(2), w2_orig(2));


  // Test on Jacobians. Remember that Eigen indexing is (row,col) while OpenCV is (col,row)
  EXPECT_FLOAT_EQ(dw0_dR(0,0), dv0_dRR.at<number_t>(0,0));
  EXPECT_FLOAT_EQ(dw0_dR(0,1), dv0_dRR.at<number_t>(1,0));
  EXPECT_FLOAT_EQ(dw0_dR(0,2), dv0_dRR.at<number_t>(2,0));
  EXPECT_FLOAT_EQ(dw0_dR(0,3), dv0_dRR.at<number_t>(3,0));
  EXPECT_FLOAT_EQ(dw0_dR(0,4), dv0_dRR.at<number_t>(4,0));
  EXPECT_FLOAT_EQ(dw0_dR(0,5), dv0_dRR.at<number_t>(5,0));
  EXPECT_FLOAT_EQ(dw0_dR(0,6), dv0_dRR.at<number_t>(6,0));
  EXPECT_FLOAT_EQ(dw0_dR(0,7), dv0_dRR.at<number_t>(7,0));
  EXPECT_FLOAT_EQ(dw0_dR(0,8), dv0_dRR.at<number_t>(8,0));
  EXPECT_FLOAT_EQ(dw0_dR(1,0), dv0_dRR.at<number_t>(0,1));
  EXPECT_FLOAT_EQ(dw0_dR(1,1), dv0_dRR.at<number_t>(1,1));
  EXPECT_FLOAT_EQ(dw0_dR(1,2), dv0_dRR.at<number_t>(2,1));
  EXPECT_FLOAT_EQ(dw0_dR(1,3), dv0_dRR.at<number_t>(3,1));
  EXPECT_FLOAT_EQ(dw0_dR(1,4), dv0_dRR.at<number_t>(4,1));
  EXPECT_FLOAT_EQ(dw0_dR(1,5), dv0_dRR.at<number_t>(5,1));
  EXPECT_FLOAT_EQ(dw0_dR(1,6), dv0_dRR.at<number_t>(6,1));
  EXPECT_FLOAT_EQ(dw0_dR(1,7), dv0_dRR.at<number_t>(7,1));
  EXPECT_FLOAT_EQ(dw0_dR(1,8), dv0_dRR.at<number_t>(8,1));
  EXPECT_FLOAT_EQ(dw0_dR(2,0), dv0_dRR.at<number_t>(0,2));
  EXPECT_FLOAT_EQ(dw0_dR(2,1), dv0_dRR.at<number_t>(1,2));
  EXPECT_FLOAT_EQ(dw0_dR(2,2), dv0_dRR.at<number_t>(2,2));
  EXPECT_FLOAT_EQ(dw0_dR(2,3), dv0_dRR.at<number_t>(3,2));
  EXPECT_FLOAT_EQ(dw0_dR(2,4), dv0_dRR.at<number_t>(4,2));
  EXPECT_FLOAT_EQ(dw0_dR(2,5), dv0_dRR.at<number_t>(5,2));
  EXPECT_FLOAT_EQ(dw0_dR(2,6), dv0_dRR.at<number_t>(6,2));
  EXPECT_FLOAT_EQ(dw0_dR(2,7), dv0_dRR.at<number_t>(7,2));
  EXPECT_FLOAT_EQ(dw0_dR(2,8), dv0_dRR.at<number_t>(8,2));


  EXPECT_FLOAT_EQ(dw1_dR(0,0), dv1_dRR.at<number_t>(0,0));
  EXPECT_FLOAT_EQ(dw1_dR(0,1), dv1_dRR.at<number_t>(1,0));
  EXPECT_FLOAT_EQ(dw1_dR(0,2), dv1_dRR.at<number_t>(2,0));
  EXPECT_FLOAT_EQ(dw1_dR(0,3), dv1_dRR.at<number_t>(3,0));
  EXPECT_FLOAT_EQ(dw1_dR(0,4), dv1_dRR.at<number_t>(4,0));
  EXPECT_FLOAT_EQ(dw1_dR(0,5), dv1_dRR.at<number_t>(5,0));
  EXPECT_FLOAT_EQ(dw1_dR(0,6), dv1_dRR.at<number_t>(6,0));
  EXPECT_FLOAT_EQ(dw1_dR(0,7), dv1_dRR.at<number_t>(7,0));
  EXPECT_FLOAT_EQ(dw1_dR(0,8), dv1_dRR.at<number_t>(8,0));
  EXPECT_FLOAT_EQ(dw1_dR(1,0), dv1_dRR.at<number_t>(0,1));
  EXPECT_FLOAT_EQ(dw1_dR(1,1), dv1_dRR.at<number_t>(1,1));
  EXPECT_FLOAT_EQ(dw1_dR(1,2), dv1_dRR.at<number_t>(2,1));
  EXPECT_FLOAT_EQ(dw1_dR(1,3), dv1_dRR.at<number_t>(3,1));
  EXPECT_FLOAT_EQ(dw1_dR(1,4), dv1_dRR.at<number_t>(4,1));
  EXPECT_FLOAT_EQ(dw1_dR(1,5), dv1_dRR.at<number_t>(5,1));
  EXPECT_FLOAT_EQ(dw1_dR(1,6), dv1_dRR.at<number_t>(6,1));
  EXPECT_FLOAT_EQ(dw1_dR(1,7), dv1_dRR.at<number_t>(7,1));
  EXPECT_FLOAT_EQ(dw1_dR(1,8), dv1_dRR.at<number_t>(8,1));
  EXPECT_FLOAT_EQ(dw1_dR(2,0), dv1_dRR.at<number_t>(0,2));
  EXPECT_FLOAT_EQ(dw1_dR(2,1), dv1_dRR.at<number_t>(1,2));
  EXPECT_FLOAT_EQ(dw1_dR(2,2), dv1_dRR.at<number_t>(2,2));
  EXPECT_FLOAT_EQ(dw1_dR(2,3), dv1_dRR.at<number_t>(3,2));
  EXPECT_FLOAT_EQ(dw1_dR(2,4), dv1_dRR.at<number_t>(4,2));
  EXPECT_FLOAT_EQ(dw1_dR(2,5), dv1_dRR.at<number_t>(5,2));
  EXPECT_FLOAT_EQ(dw1_dR(2,6), dv1_dRR.at<number_t>(6,2));
  EXPECT_FLOAT_EQ(dw1_dR(2,7), dv1_dRR.at<number_t>(7,2));
  EXPECT_FLOAT_EQ(dw1_dR(2,8), dv1_dRR.at<number_t>(8,2));

  EXPECT_FLOAT_EQ(dw2_dR(0,0), dv2_dRR.at<number_t>(0,0));
  EXPECT_FLOAT_EQ(dw2_dR(0,1), dv2_dRR.at<number_t>(1,0));
  EXPECT_FLOAT_EQ(dw2_dR(0,2), dv2_dRR.at<number_t>(2,0));
  EXPECT_FLOAT_EQ(dw2_dR(0,3), dv2_dRR.at<number_t>(3,0));
  EXPECT_FLOAT_EQ(dw2_dR(0,4), dv2_dRR.at<number_t>(4,0));
  EXPECT_FLOAT_EQ(dw2_dR(0,5), dv2_dRR.at<number_t>(5,0));
  EXPECT_FLOAT_EQ(dw2_dR(0,6), dv2_dRR.at<number_t>(6,0));
  EXPECT_FLOAT_EQ(dw2_dR(0,7), dv2_dRR.at<number_t>(7,0));
  EXPECT_FLOAT_EQ(dw2_dR(0,8), dv2_dRR.at<number_t>(8,0));
  EXPECT_FLOAT_EQ(dw2_dR(1,0), dv2_dRR.at<number_t>(0,1));
  EXPECT_FLOAT_EQ(dw2_dR(1,1), dv2_dRR.at<number_t>(1,1));
  EXPECT_FLOAT_EQ(dw2_dR(1,2), dv2_dRR.at<number_t>(2,1));
  EXPECT_FLOAT_EQ(dw2_dR(1,3), dv2_dRR.at<number_t>(3,1));
  EXPECT_FLOAT_EQ(dw2_dR(1,4), dv2_dRR.at<number_t>(4,1));
  EXPECT_FLOAT_EQ(dw2_dR(1,5), dv2_dRR.at<number_t>(5,1));
  EXPECT_FLOAT_EQ(dw2_dR(1,6), dv2_dRR.at<number_t>(6,1));
  EXPECT_FLOAT_EQ(dw2_dR(1,7), dv2_dRR.at<number_t>(7,1));
  EXPECT_FLOAT_EQ(dw2_dR(1,8), dv2_dRR.at<number_t>(8,1));
  EXPECT_FLOAT_EQ(dw2_dR(2,0), dv2_dRR.at<number_t>(0,2));
  EXPECT_FLOAT_EQ(dw2_dR(2,1), dv2_dRR.at<number_t>(1,2));
  EXPECT_FLOAT_EQ(dw2_dR(2,2), dv2_dRR.at<number_t>(2,2));
  EXPECT_FLOAT_EQ(dw2_dR(2,3), dv2_dRR.at<number_t>(3,2));
  EXPECT_FLOAT_EQ(dw2_dR(2,4), dv2_dRR.at<number_t>(4,2));
  EXPECT_FLOAT_EQ(dw2_dR(2,5), dv2_dRR.at<number_t>(5,2));
  EXPECT_FLOAT_EQ(dw2_dR(2,6), dv2_dRR.at<number_t>(6,2));
  EXPECT_FLOAT_EQ(dw2_dR(2,7), dv2_dRR.at<number_t>(7,2));
  EXPECT_FLOAT_EQ(dw2_dR(2,8), dv2_dRR.at<number_t>(8,2));

}