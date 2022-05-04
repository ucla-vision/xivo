#include <iostream>

#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "rodrigues.h"

using number_t = double;
const number_t eps = 1e-8;

using namespace xivo;

class MatrixDifferentialTest : public ::testing::Test {
protected:
    MatrixDifferentialTest() {
        _A1.setRandom();
        _A2.setRandom();
        _A.setRandom();

        _B1.setRandom();
        _B2.setRandom();
        _B.setRandom();
    }
    ~MatrixDifferentialTest() override {}
    void SetUp() override {}
    void TearDown() override {}

public:
    Eigen::Matrix<number_t, 3, 2> _A1;
    Eigen::Matrix<number_t, 2, 4> _A2;

    Eigen::Matrix<number_t, 4, 2> _B1;
    Eigen::Matrix<number_t, 2, 5> _B2;

    Eigen::Matrix<number_t, 3, 4> _A;
    Eigen::Matrix<number_t, 4, 5> _B;
};


TEST_F(MatrixDifferentialTest, dAB_dA) {
    // setup the differential operator
    auto diff = dAB_dA(_A, _B);

    // now let's first compute derivative of C w.r.t. each component of A
    auto C = _A * _B;
    Eigen::Matrix<number_t, 15, 12> num_diff;
    num_diff.setZero();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            auto Ap(_A);
            Ap(i, j) += eps;
            Eigen::Matrix<number_t, 3, 5> D = (Ap * _B - C) / eps;
            num_diff.col(i*4+j) = Eigen::Map<Eigen::Matrix<number_t, 15, 1>>(D.data());
        }
    }
    ASSERT_LE((diff - num_diff).norm(), 1e-3) << "inconsistent analytical & numerical derivatives";
}

// dAB_dA with A and B of expression type
TEST_F(MatrixDifferentialTest, dAB_dA_expression) {
    auto A = _A1 * _A2;
    auto B = _B1 * _B2;
    static_assert(!std::is_same<decltype(A), decltype(_A)>::value, "A and _A should have different types.");
    static_assert(!std::is_same<decltype(B), decltype(_B)>::value, "B and _B should have different types.");
    dAB_dA(A, B);
}

TEST_F(MatrixDifferentialTest, dAB_dB) {
    // setup the differential operator
    auto diff = dAB_dB(_A, _B);

    // now let's first compute derivative of C w.r.t. each component of A
    auto C = _A * _B;
    Eigen::Matrix<number_t, 15, 20> num_diff;
    num_diff.setZero();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 5; ++j) {
            auto Bp(_B);
            Bp(i, j) += eps;
            Eigen::Matrix<number_t, 3, 5> D = (_A * Bp - C) / eps;
            num_diff.col(i*5+j) = Eigen::Map<Eigen::Matrix<number_t, 15, 1>>(D.data());
        }
    }
    ASSERT_LE((diff - num_diff).norm(), 1e-3) << "inconsistent analytical & numerical derivatives";
}

TEST_F(MatrixDifferentialTest, dAB_dB_expression) {
    // dAB_dB with A and B of expression type
    auto A = _A1 * _A2;
    auto B = _B1 * _B2;
    static_assert(!std::is_same<decltype(A), decltype(_A)>::value, "A and _A should have different types.");
    static_assert(!std::is_same<decltype(B), decltype(_B)>::value, "B and _B should have different types.");
    dAB_dB(A, B);
}

TEST_F(MatrixDifferentialTest, dhat) {
    Eigen::Matrix<number_t, 3, 1> u;
    u.setRandom();
    Eigen::Matrix<number_t, 9, 1> product = dhat(u) * u;
    auto a = Eigen::Map<Eigen::Matrix<number_t, 3, 3>>(product.data());
    // c=a.T=-a

    a.transposeInPlace();

    auto c = a.transpose();
    auto b = hat(u);
    ASSERT_LE((a-b).norm(), 1e-10);
    ASSERT_LE((c+b).norm(), 1e-10);
}

TEST_F(MatrixDifferentialTest, dhat_expression) {
    Eigen::Matrix<number_t, 3, 1> u;
    u.setRandom();
    static_assert(!std::is_same<decltype(u.head<3>()), decltype(u)>::value, "u.head<3>() and u should have different types.");
    dhat(u.head<3>());
}

TEST_F(MatrixDifferentialTest, dAt_dA) {
    Eigen::Matrix<number_t, 4, 4> A;
    A.setRandom();
    auto At = A.transpose();
    Eigen::Matrix<number_t, 16, 1> D = dAt_dA(A) * Eigen::Map<Eigen::Matrix<number_t, 16, 1>>(A.data());
    auto at = Eigen::Map<Eigen::Matrix<number_t, 4, 4>>(D.data());
    ASSERT_LE((At-at).norm(), 1e-10);
}

TEST_F(MatrixDifferentialTest, rodrigues) {
    Eigen::Matrix<number_t, 3, 1> w;
    w.setRandom();
    Eigen::Matrix<number_t, 9, 3> dR_dw;
    auto R = rodrigues(w, &dR_dw);
    auto RRt = R * R.transpose();
    ASSERT_LE((Eigen::Matrix<number_t, 3, 3>::Identity() - RRt).norm(), 1e-5);

    Eigen::Matrix<number_t, 9, 3> num_dR_dw;
    num_dR_dw.setZero();
    for (int i = 0; i < 3; ++i) {
        Eigen::Matrix<number_t, 3, 1> wp = w;
        wp(i) += eps;
        Eigen::Matrix<number_t, 3, 3> dR_dw_i = ((rodrigues(wp) - R) / eps).transpose();
        num_dR_dw.col(i) = Eigen::Map<Eigen::Matrix<number_t, 9, 1>>(dR_dw_i.data());
    }

    ASSERT_LE((num_dR_dw - dR_dw).norm(), 1e-5);
}

// make sure rodrigues accepts induced types
TEST_F(MatrixDifferentialTest, rodrigues_expression) {
    Eigen::Matrix<number_t, 3, 1> w;
    w.setRandom();
    static_assert(!std::is_same<decltype(w.head<3>()), decltype(w)>::value,
        "w.head<3>() should have type different from Eigen::Matrix<number_t, 3, 1>.");
    rodrigues(w.head<3>());
}

TEST_F(MatrixDifferentialTest, rodrigues_small_angle) {
    Eigen::Matrix<number_t, 3, 1> w;
    w.setRandom();
    w /= 1e10;
    Eigen::Matrix<number_t, 9, 3> dR_dw;
    auto R = rodrigues(w, &dR_dw);
    auto RRt = R * R.transpose();
    ASSERT_LE((Eigen::Matrix<number_t, 3, 3>::Identity() - RRt).norm(), 1e-5);

    Eigen::Matrix<number_t, 9, 3> num_dR_dw;
    num_dR_dw.setZero();
    for (int i = 0; i < 3; ++i) {
        Eigen::Matrix<number_t, 3, 1> wp = w;
        wp(i) += eps;
        Eigen::Matrix<number_t, 3, 3> dR_dw_i = ((rodrigues(wp) - R) / eps).transpose();
        num_dR_dw.col(i) = Eigen::Map<Eigen::Matrix<number_t, 9, 1>>(dR_dw_i.data());
    }

    ASSERT_LE((num_dR_dw - dR_dw).norm(), 1e-5);
}


TEST_F(MatrixDifferentialTest, invrodrigues) {
    Eigen::Matrix<number_t, 3, 1> w;
    w.setRandom();
    Eigen::Matrix<number_t, 3, 3> R = rodrigues(w);

    Eigen::Matrix<number_t, 3, 9> dw_dR;
    w = invrodrigues(R, &dw_dR);

    Eigen::Matrix<number_t, 3, 9> num_dw_dR;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Eigen::Matrix<number_t, 3, 3> Rp(R);
            // In theory, rotation matrix + delta is not necessarily a rotation matrix.
            Rp(i, j) += eps;
            Eigen::Matrix<number_t, 3, 1> wp = invrodrigues(Rp);
            num_dw_dR.col(i*3+j) = (wp - w) / eps;
        }
    }

    ASSERT_LE((dw_dR - num_dw_dR).norm(), 1e-5);
}

TEST_F(MatrixDifferentialTest, invrodrigues_expression) {
    Eigen::Matrix<number_t, 3, 1> w;
    w.setRandom();
    Eigen::Matrix<number_t, 3, 3> R = rodrigues(w);
    static_assert(!std::is_same<decltype(R.block<3, 3>(0, 0)), decltype(R)>::value,
        "R.block<3, 3>(0, 0) and R should have different types.");
    invrodrigues(R.block<3, 3>(0, 0));
}


TEST_F(MatrixDifferentialTest, invrodrigues_small_angle) {
    Eigen::Matrix<number_t, 3, 1> w;
    w.setIdentity();
    w *= eps;
    Eigen::Matrix<number_t, 3, 3> R = rodrigues(w);

    Eigen::Matrix<number_t, 3, 9> dw_dR;
    w = invrodrigues(R, &dw_dR);

    Eigen::Matrix<number_t, 3, 9> num_dw_dR;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Eigen::Matrix<number_t, 3, 3> Rp(R);
            // In theory, rotation matrix + delta is not necessarily a rotation matrix.
            Rp(i, j) += eps;
            Eigen::Matrix<number_t, 3, 1> wp = invrodrigues(Rp);
            num_dw_dR.col(i*3+j) = (wp - w) / eps;
        }
    }

    ASSERT_LE((dw_dR - num_dw_dR).norm(), 1e-5);
}


TEST_F(MatrixDifferentialTest, dRu_dw) {
    Eigen::Matrix<number_t, 3, 1> w{1.1, 2.2, 3.3};
    //Eigen::Matrix<number_t, 3, 1> w{-0.579252,   -1.1585,  -1.73776};
    //Eigen::Matrix<number_t, 3, 1> u{-3.0, 4.5, -12.2};
    Eigen::Matrix<number_t, 3, 1> accel{-9.57653808594, 0.134033203125, 1.72415161133};
    Eigen::Matrix<number_t, 3, 1> ba{1e-3, 2.5e-3, -5.0e-4};
    Eigen::Matrix<number_t, 3, 3> Ta, Ka;
    Ta << 1, 0.00533542, 0.00268388,
            0,          1, -0.0107169,
            0,          0,          1;
    Ka << 0.997708, 0.0, 0.0,
            0.0, 0.997608, 0.0,
            0.0, 0.0, 0.987496;
    Eigen::Matrix<number_t, 3, 3> Ca = Ta * Ka;

    Eigen::Matrix<number_t, 3, 1> u = Ca * accel - ba;
    std::cout << "Wsb: " << w.transpose() << std::endl;
    std::cout << "u: " << u.transpose() << std::endl;
    std::cout << "Rsb: " << std::endl << rodrigues(w) << std::endl;

    Eigen::Matrix<number_t, 3, 1> y0 = rodrigues(w) * u;
    Eigen::Matrix<number_t, 3, 3> jac = dRu_dw(w, u);

    Eigen::Matrix<number_t, 3, 3> num_jac;

    for (int j = 0; j < 3; j++) {
        Eigen::Matrix<number_t, 3, 1> wp(w);
        wp(j) += eps;
        Eigen::Matrix<number_t, 3, 1> y1 = rodrigues(wp) * u;

        num_jac.col(j) = (y1 - y0) / eps;
    }

    ASSERT_LE((jac - num_jac).norm(), 1e-5);
    std::cout << "jac" << std::endl << jac << std::endl;
    std::cout << "num_jac" << std::endl << num_jac << std::endl;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(jac(i,j), num_jac(i,j), 1e-5) <<
            "dRu_dw test error at i=" << i << " and j=" << j;
        }
    }
}


TEST_F(MatrixDifferentialTest, dRu_dw_small_angle) {
    Eigen::Matrix<number_t, 3, 1> w{1e-8, 0, 0};
    Eigen::Matrix<number_t, 3, 1> u;
    u.setRandom();

    Eigen::Matrix<number_t, 3, 1> y0 = rodrigues(w) * u;
    Eigen::Matrix<number_t, 3, 3> jac = dRu_dw(w, u);
    Eigen::Matrix<number_t, 3, 3> num_jac;

    for (int j = 0; j < 3; j++) {
        Eigen::Matrix<number_t, 3, 1> wp(w);
        wp(j) += eps;
        Eigen::Matrix<number_t, 3, 1> y1 = rodrigues(wp) * u;

        num_jac.col(j) = (y1 - y0) / eps;
    }

    ASSERT_LE((jac - num_jac).norm(), 1e-5);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(jac(i,j), num_jac(i,j), 1e-5) <<
            "small angle dRu_dw test error at i=" << i << " and j=" << j;
        }
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


TEST_F(MatrixDifferentialTest, dRu_dw2) {
    Eigen::Matrix<number_t, 3, 1> wg{0.0123478, -1.301, 0};
    Eigen::Matrix<number_t, 3, 1> g{0.0, 0.0, -9.8};

    Eigen::Matrix<number_t, 3, 1> u = g;
    std::cout << "Wsg: " << wg.transpose() << std::endl;
    std::cout << "u: " << u.transpose() << std::endl;
    std::cout << "Rsg: " << std::endl << rodrigues(wg) << std::endl;

    Eigen::Matrix<number_t, 3, 1> y0 = rodrigues(wg) * u;
    Eigen::Matrix<number_t, 3, 3> jac = dRu_dw(wg, u);

    Eigen::Matrix<number_t, 3, 3> num_jac;
    num_jac.setZero();

    for (int j = 0; j < 3; j++) {
        Eigen::Matrix<number_t, 3, 1> wp(wg);
        wp(j) += eps;
        Eigen::Matrix<number_t, 3, 1> y1 = rodrigues(wp) * u;

        num_jac.col(j) = (y1 - y0) / eps;
    }

    ASSERT_LE((jac - num_jac).norm(), 1e-5);
    std::cout << "jac" << std::endl << jac << std::endl;
    std::cout << "num_jac" << std::endl << num_jac << std::endl;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(jac(i,j), num_jac(i,j), 1e-5) <<
            "dRu_dw2 test error at i=" << i << " and j=" << j;
        }
    }
}