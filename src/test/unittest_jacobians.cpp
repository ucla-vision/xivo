#include <gtest/gtest.h>
#include <Eigen/Geometry>

#include "alias.h"

#define private public
#include "feature.h"


using namespace Eigen;
using namespace xivo;

/* Class to check dxc(now)_d[everything else]. We wil */
class InstateJacobiansTest : public ::testing::Test {
  protected:
    void SetUp() override {
        f = Feature::Create(25, 46);
        delta = 1e-6;
        tol = 1e-6;

        // IMU measurement
        gyro = Vec3::Random();

        // Bogus values for xc(ref) = [2m, 3m, 10m],
#ifdef USE_INVDEPTH
        f -> x_ << 2, 3, 0.2;
#else
        f -> x_ << 2, 3, log(10);
#endif

        // Set nominal and error variables to random values
        Rr_nom = Quaternion<number_t, AutoAlign>::UnitRandom().toRotationMatrix();
        Tr_nom = Vec3::Random();
        Rsb_nom = Vec3::Random();
        Tsb_nom = Vec3::Random();
        Rbc_nom = Vec3::Random();
        Tbc_nom = Vec3::Random();
        Cg_nom = Mat3::Identity();
        bg_nom << 0, 0, 0;
        td_nom = 0.005;
        Vsb_nom = Vec3::Random();

        Wr_err = Vec3::Random();
        Tr_err = Vec3::Random();
        Wsb_err = Vec3::Random();
        Tsb_err = Vec3::Random();
        Wbc_err = Vec3::Random();
        Tbc_err = Vec3::Random();
        Cg_err = Mat3::Random();
        bg_err = Vec3::Random();
        td_err = rand();

        // Compute the analytic Jacobians
        f->ComputeJacobian(Rsb_nom, Tsb_nom, Rbc_nom, Tbc_nom, gyro, 
                           Cg_nom, bg_nom, Vsb_nom, td_nom);
    }

    Vec3 XcFormula() {
        Vec3 Xc = f->Xc();
        Vec3 Xs = Rr*(Rbc*Xc + Tbc) + Tr;

        Mat3 Rbc_t = Rbc.transpose();
        Mat3 Rsb_t = Rsb.transpose();
        Vec3 Xcn = Rbc_t*(Rsb_t*(Xs - Tsb) - Tbc);

        return Xcn;
    }

    void CreateRealStateValues() {
        Mat3 I3 = Mat3::Identity();

        Rr = Rr_nom*(I3 + hat(Wr_err));
        Tr = Tr_nom + Tr_err;
        Rsb = Rsb_nom*(I3 + hat(Wsb_err));
        Tsb = Tsb_nom + Tsb_err;
        Rbc = Rbc_nom*(I3 + hat(Wbc_err));
        Tbc = Tbc_nom + Tbc_err;
        Cg = Cg_nom + Cg_err;
        bg = bg_nom + bg_err;
        td = td_nom + td_err;
    }

    // Feature Object
    FeaturePtr f;

    // A value of xc(new) that contains all the errors.
    Vec3 xc;

    // Fake IMU measurement
    Vec3 gyro;

    // numerical tolerance
    number_t tol;

    // Real values (= nominal + error)
    Mat3 Rr;
    Vec3 Tr;
    Mat3 Rsb;
    Vec3 Tsb;
    Mat3 Rbc;
    Vec3 Tbc;
    Mat3 Cg;
    Vec3 bg;
    number_t td;

    // Nominal state variables containing placeholder values
    Mat3 Rr_nom;
    Vec3 Tr_nom;
    Mat3 Rsb_nom;
    Vec3 Tsb_nom;
    Mat3 Rbc_nom;
    Vec3 Tbc_nom;
    Mat3 Cg_nom;
    Vec3 bg_nom;
    number_t td_nom;
    Vec3 Vsb_nom;

    // Error variables containing placeholder values
    Vec3 Wr_err;
    Vec3 Tr_err;
    Vec3 Wsb_err;
    Vec3 Tsb_err;
    Vec3 Wbc_err;
    Vec3 Tbc_err;
    Mat3 Cg_err;
    Vec3 bg_err;
    number_t td_err;

    // finite difference
    number_t delta;
};


TEST_F(InstateJacobiansTest, Wr) {
    CreateRealStateValues();
    Vec3 Xcn0 = XcFormula();

    Wr_err(0) += delta;
    CreateRealStateValues();
    Vec3 Xcn1_0 = XcFormula();

    Wr_err(0) -= delta;
    Wr_err(1) += delta;
    CreateRealStateValues();
    Vec3 Xcn1_1 = XcFormula();

    Wr_err(1) -= delta;
    Wr_err(2) += delta;
    CreateRealStateValues();
    Vec3 Xcn1_2 = XcFormula();

    Vec3 dXcn_dWr0 = (Xcn1_0 - Xcn0) / delta;
    Vec3 dXcn_dWr1 = (Xcn1_1 - Xcn0) / delta;
    Vec3 dXcn_dWr2 = (Xcn1_2 - Xcn0) / delta;

    EXPECT_NEAR(dXcn_dWr0(0), f->cache_.dXcn_dWr(0,0), tol);
    EXPECT_NEAR(dXcn_dWr0(1), f->cache_.dXcn_dWr(1,0), tol);
    EXPECT_NEAR(dXcn_dWr0(2), f->cache_.dXcn_dWr(2,0), tol);
    
    EXPECT_NEAR(dXcn_dWr1(0), f->cache_.dXcn_dWr(0,1), tol);
    EXPECT_NEAR(dXcn_dWr1(1), f->cache_.dXcn_dWr(1,1), tol);
    EXPECT_NEAR(dXcn_dWr1(2), f->cache_.dXcn_dWr(2,1), tol);

    EXPECT_NEAR(dXcn_dWr2(0), f->cache_.dXcn_dWr(0,2), tol);
    EXPECT_NEAR(dXcn_dWr2(1), f->cache_.dXcn_dWr(1,2), tol);
    EXPECT_NEAR(dXcn_dWr2(2), f->cache_.dXcn_dWr(2,2), tol);
}