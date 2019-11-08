#include "unittest_helpers.h"
#include "gtest/gtest.h"


void CheckVectorEquality(VecX v1, VecX v2, number_t tol)
{
    EXPECT_EQ(v1.size(), v2.size());
    for (int i=0; i<v1.size(); i++) {
        EXPECT_NEAR(v1(i), v2(i), tol);
    }
}


void CheckMatrixEquality(MatX M1, MatX M2, number_t tol) {
    EXPECT_EQ(M1.rows(), M2.rows());
    EXPECT_EQ(M1.cols(), M2.cols());

    for (int i=0; i<M1.rows(); i++) {
        for (int j=0; j<M1.cols(); j++) {
            EXPECT_NEAR(M1(i,j), M2(i,j), tol);
        }
    }
}

void CheckVecZero(VecX v, number_t tol) {
    for (int i=0; i<v.size(); i++) {
        EXPECT_NEAR(v(i), 0.0, tol);
    }
}

void CheckMatrixZero(MatX M, number_t tol) {
    for (int i=0; i<M.rows(); i++) {
        for (int j=0; j<M.cols(); j++) {
            EXPECT_NEAR(M(i,j), 0.0, tol);
        }
    }
}