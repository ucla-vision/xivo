#include "helpers.cpp"
#include "gtest/gtest.h"

using namespace xivo;


// These tests come from here:
// https://www.math.usm.edu/lambers/mat610/sum10/lecture9.pdf 
// Can double-check using 'planerot' function in MATLAB.
// Note that we use the convention in Golub & Loan
// tests are accurate up to a sign.
TEST(NumericalLinearAlgebra, GivensSub) {
    double tol = 5e-4;

    Vec2 v1(0.9134, 0.6324);
    Mat2 G1 = givens(v1(0), v1(1));
    Vec2 y1 = G1.transpose() * v1;
    EXPECT_NEAR(abs(y1(0)), 1.1110, tol);
    EXPECT_NEAR(y1(1), 0.0, tol);
    EXPECT_NEAR(abs(G1(0,0)), 0.8222, tol);
    EXPECT_NEAR(abs(G1(0,1)), 0.5692, tol);
    EXPECT_NEAR(abs(G1(1,0)), 0.5692, tol);
    EXPECT_NEAR(abs(G1(1,1)), 0.8222, tol);

    Vec2 v2(0.1270, 1.1109);
    Mat2 G2 = givens(v2(0), v2(1));
    Vec2 y2 = G2.transpose() * v2;
    EXPECT_NEAR(abs(y2(0)), 1.1181, tol);
    EXPECT_NEAR(y2(1), 0.0, tol);
    EXPECT_NEAR(abs(G2(0,0)), 0.1136, tol);
    EXPECT_NEAR(abs(G2(0,1)), 0.9935, tol);
    EXPECT_NEAR(abs(G2(1,0)), 0.9935, tol);
    EXPECT_NEAR(abs(G2(1,1)), 0.1136, tol);
}


TEST(NumericalLinearAlgebra, SlowGivens) {


}