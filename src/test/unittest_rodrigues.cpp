#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>

#include "alias.h"
#include "rodrigues.h"
#include "timer.h"

using namespace Eigen;
using namespace xivo;


// Test Rodrigues and derivatives against OpenCV implementation for correctness
TEST(Rodrigues, Correctness) {

  double tol = 1e-6;

  // Test vectors
  Vec3 w0(0.5, 0.5, 0.7);
  Vec3 w1(0.1, 0.2, -0.1);

  Timer t;

  // Output Jacobians
  Eigen::Matrix<number_t, 9, 3> dR_dw0;
  Eigen::Matrix<number_t, 9, 3> dR_dw1;

  // Our implementation
  t.Tick("Ours");
  Mat3 R0 = rodrigues(w0, &dR_dw0);
  t.Tock("Ours");
  t.Tick("Ours");
  Mat3 R1 = rodrigues(w1, &dR_dw1);
  t.Tock("Ours");

  // OpenCV implementation
  t.Tick("OpenCV");
  cv::Vec3d v0( w0(0), w0(1), w0(2) );
  cv::Matx33d RR0;
  cv::Mat_<number_t> dRR_dv0(3, 9, 0.0); // OpenCV does (cols, rows) to be like images
  cv::Rodrigues(v0, RR0, dRR_dv0);
  t.Tock("OpenCV");

  t.Tick("OpenCV");
  cv::Vec3d v1( w1(0), w1(1), w1(2) );
  cv::Matx33d RR1;
  cv::Mat_<number_t> dRR_dv1(3, 9, 0.0);
  cv::Rodrigues(v1, RR1, dRR_dv1);
  t.Tock("OpenCV");

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


  // Print to test speed
  std::cout << t << std::endl;
}