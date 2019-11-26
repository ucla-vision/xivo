#include "helpers.cpp"
#include "unittest_helpers.h"
#include "Eigen/SVD"
#include "gtest/gtest.h"

using namespace Eigen;
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


TEST(NumericalLinearAlgebra, SlowAndFastGivensMatch) {
    number_t tol = 1e-4;

    int M = 4;
    VecX r, r2;
    MatX Hf, Hx, Hf2, Hx2;
    r = MatX::Random(2 * M, 1);
    Hf = MatX::Random(2 * M, 3);
    Hx = MatX::Random(2 * M, 5);

    // For comparing Givens and SlowGivens
    r2 = r;
    Hf2 = Hf;
    Hx2 = Hx;

    std::cout << "===== Before givens =====\n";
    std::cout << "r=\n";
    std::cout << r.transpose() << std::endl;
    std::cout << "Hf=\n";
    std::cout << Hf << std::endl;
    std::cout << "Hx=\n";
    std::cout << Hx << std::endl;

    int effective_rows = Givens(r, Hx, Hf);
    std::cout << "===== After givens =====\n";
    std::cout << "r=\n";
    std::cout << r.transpose() << std::endl;
    std::cout << "Hf=\n";
    std::cout << Hf << std::endl;
    std::cout << "Hx=\n";
    std::cout << Hx << std::endl;
    std::cout << "Effective Rows: " << effective_rows << std::endl;
    JacobiSVD<MatX> svd(Hx);
    std::cout << "Singular values are: " << std::endl << svd.singularValues() << std::endl;

    MatX A;
    int effective_rows2 = SlowGivens(Hf2, Hx2, A);
    VecX r2_after = A.transpose() * r2;
    MatX Hf2_after = A.transpose() * Hf2;
    std::cout << "===== After Slow Givens =====\n";
    std::cout << "r=\n";
    std::cout << r2_after.transpose() << std::endl;
    std::cout << "Hf=\n";
    std::cout << Hf2_after << std::endl;
    std::cout << "Hx=\n";
    std::cout << Hx2 << std::endl;

    EXPECT_EQ(effective_rows, effective_rows2);
    CheckMatrixEquality(Hf.block(0,0,effective_rows, 3), Hf2_after, tol);
    CheckMatrixZero(Hf, tol);
    CheckMatrixZero(Hf2_after, tol);
    CheckVectorEquality(r.head(effective_rows), r2_after, tol);
    CheckMatrixEquality(Hx.block(0,0,effective_rows,5), Hx2, tol);
}


TEST(NumericalLinearAlgebra, QR) {
    int N = 4;  // state size
    int M = 8;  // measurement size
    VecX r;
    MatX Hf, Hx;
    r = MatX::Random(M, 1);
    Hx = MatX::Random(M, N);

    std::cout << "r=\n" << r.transpose() << std::endl;
    std::cout << "Hx=\n" << Hx << std::endl;
    int rows = QR(r, Hx);
    std::cout << "Effective rows: " << rows << std::endl;
    std::cout << "===== After givens =====\n";
    std::cout << "r=\n";
    std::cout << r.head(rows).transpose() << std::endl;
    std::cout << "TH=\n";
    std::cout << Hx.topRows(rows) << std::endl;

}