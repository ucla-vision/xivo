#include <gtest/gtest.h>
#include <Eigen/Geometry>

#define private public

#include "alias.h"
#include "mm.h"
#include "group.h"
#include "graph.h"

#include "unittest_helpers.h"

#include "feature.h"


using namespace Eigen;
using namespace xivo;

/* Class to check dxc(now)_d[everything else]. We wil */
class InstateJacobiansTest : public ::testing::Test {
  protected:
    void SetUp() override {

        // Create feature object and testing parameters
        MemoryManager::Create(256, 128);
        auto cfg_ = LoadJson("cfg/phab.json");
        Camera::Create(cfg_["camera_cfg"]);
        delta = 1e-6;
        tol = 1e-6;

        // IMU measurement
        gyro = Vec3::Random();


        // Set nominal and error variables to random values
        Rr_nom = RandomTransformationMatrix();
        Tr_nom = Vec3::Random();
        Rsb_nom = RandomTransformationMatrix();
        Tsb_nom = Vec3::Random();
        Rbc_nom = RandomTransformationMatrix();
        Tbc_nom = Vec3::Random();
        Cg_nom = Mat3::Identity();
        bg_nom << 0, 0, 0;
        td_nom = 0.005;
        Vsb_nom = Vec3::Random();

        Wr_err = Vec3::Zero();
        Tr_err = Vec3::Zero();
        Wsb_err = Vec3::Zero();
        Tsb_err = Vec3::Zero();
        Wbc_err = Vec3::Zero();
        Tbc_err = Vec3::Zero();
        Cg_err = Mat3::Zero();
        bg_err = Vec3::Zero();
        td_err = 0.0;

        // Set reference Rr and Tr for the feature
        f = Feature::Create(25, 46);
        // Bogus values for xc(ref) = [2m, 3m, 10m],
#ifdef USE_INVDEPTH
        f -> x_ << 2, 3, (1/10.0);
#else
        f -> x_ << 2, 3, log(10);
#endif
        group = Group::Create(SO3(Rr_nom), Tr_nom);
        f->ref_ = group;

        // Compute the analytic Jacobians and nominal states
        ComputeNominalStates();
        f->ComputeJacobian(Rsb_nom, Tsb_nom, Rbc_nom, Tbc_nom, gyro, 
                           Cg_nom, bg_nom, Vsb_nom, td_nom);
    }

    Vec3 ComputeXcn() {
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


        Vec3 Xc = f->Xc();
        Vec3 Xs = Rr*(Rbc*Xc + Tbc) + Tr;

        Mat3 Rbc_t = Rbc.transpose();
        Mat3 Rsb_t = Rsb.transpose();
        Vec3 Xcn = Rbc_t*(Rsb_t*(Xs - Tsb) - Tbc);

        return Xcn;
    }

    void ComputeNominalStates() {
        Mat3 Rsb_nom_t = Rsb_nom.transpose();
        Mat3 Rbc_nom_t = Rbc_nom.transpose();

        Xc_nom = f->Xc(nullptr);
        Xs_nom = Rr_nom * (Rbc_nom * Xc_nom + Tbc_nom) + Tr_nom;
        Xcn_nom = Rbc_nom_t*(Rsb_nom_t*(Xs_nom - Tsb_nom) - Tbc_nom);
    }

    // Feature Object and Memory Manager
    MemoryManagerPtr mm;
    CameraPtr cam;
    GroupPtr group;
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

    // Values to save for debugging
    Vec3 Xc_nom;
    Vec3 Xs_nom;
    Vec3 Xcn_nom;
};


TEST_F(InstateJacobiansTest, NominalStates) {
    EXPECT_NEAR(Xc_nom(0), f->cache_.Xc(0), tol);
    EXPECT_NEAR(Xc_nom(1), f->cache_.Xc(1), tol);
    EXPECT_NEAR(Xc_nom(2), f->cache_.Xc(2), tol);

    EXPECT_NEAR(Xs_nom(0), f->cache_.Xs(0), tol);
    EXPECT_NEAR(Xs_nom(1), f->cache_.Xs(1), tol);
    EXPECT_NEAR(Xs_nom(2), f->cache_.Xs(2), tol);

    EXPECT_NEAR(Xcn_nom(0), f->cache_.Xcn(0), tol);
    EXPECT_NEAR(Xcn_nom(1), f->cache_.Xcn(1), tol);
    EXPECT_NEAR(Xcn_nom(2), f->cache_.Xcn(2), tol);
}


TEST_F(InstateJacobiansTest, Wr) {
    Vec3 Xcn0 = ComputeXcn();

    Wr_err(0) = delta;
    Vec3 Xcn1_0 = ComputeXcn();
    Wr_err(0) = 0;

    Wr_err(1) = delta;
    Vec3 Xcn1_1 = ComputeXcn();
    Wr_err(1) = 0;

    Wr_err(2) = delta;
    Vec3 Xcn1_2 = ComputeXcn();
    Wr_err(2) = 0;

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


TEST_F(InstateJacobiansTest, Tr) {
    Vec3 Xcn0 = ComputeXcn();

    Tr_err(0) = delta;
    Vec3 Xcn1_0 = ComputeXcn();
    Tr_err(0) = 0;

    Tr_err(1) = delta;
    Vec3 Xcn1_1 = ComputeXcn();
    Tr_err(1) = 0;

    Tr_err(2) = delta;
    Vec3 Xcn1_2 = ComputeXcn();
    Tr_err(2) = 0;

    Vec3 dXcn_dTr0 = (Xcn1_0 - Xcn0) / delta;
    Vec3 dXcn_dTr1 = (Xcn1_1 - Xcn0) / delta;
    Vec3 dXcn_dTr2 = (Xcn1_2 - Xcn0) / delta;

    EXPECT_NEAR(dXcn_dTr0(0), f->cache_.dXcn_dTr(0,0), tol);
    EXPECT_NEAR(dXcn_dTr0(1), f->cache_.dXcn_dTr(1,0), tol);
    EXPECT_NEAR(dXcn_dTr0(2), f->cache_.dXcn_dTr(2,0), tol);
    
    EXPECT_NEAR(dXcn_dTr1(0), f->cache_.dXcn_dTr(0,1), tol);
    EXPECT_NEAR(dXcn_dTr1(1), f->cache_.dXcn_dTr(1,1), tol);
    EXPECT_NEAR(dXcn_dTr1(2), f->cache_.dXcn_dTr(2,1), tol);

    EXPECT_NEAR(dXcn_dTr2(0), f->cache_.dXcn_dTr(0,2), tol);
    EXPECT_NEAR(dXcn_dTr2(1), f->cache_.dXcn_dTr(1,2), tol);
    EXPECT_NEAR(dXcn_dTr2(2), f->cache_.dXcn_dTr(2,2), tol);
}


TEST_F(InstateJacobiansTest, Wsb) {
    Vec3 Xcn0 = ComputeXcn();

    Wsb_err(0) = delta;
    Vec3 Xcn1_0 = ComputeXcn();
    Wsb_err(0) = 0;

    Wsb_err(1) = delta;
    Vec3 Xcn1_1 = ComputeXcn();
    Wsb_err(1) = 0;

    Wsb_err(2) = delta;
    Vec3 Xcn1_2 = ComputeXcn();
    Wsb_err(2) = 0;

    Vec3 dXcn_dWsb0 = (Xcn1_0 - Xcn0) / delta;
    Vec3 dXcn_dWsb1 = (Xcn1_1 - Xcn0) / delta;
    Vec3 dXcn_dWsb2 = (Xcn1_2 - Xcn0) / delta;

    EXPECT_NEAR(dXcn_dWsb0(0), f->cache_.dXcn_dWsb(0,0), tol);
    EXPECT_NEAR(dXcn_dWsb0(1), f->cache_.dXcn_dWsb(1,0), tol);
    EXPECT_NEAR(dXcn_dWsb0(2), f->cache_.dXcn_dWsb(2,0), tol);
    
    EXPECT_NEAR(dXcn_dWsb1(0), f->cache_.dXcn_dWsb(0,1), tol);
    EXPECT_NEAR(dXcn_dWsb1(1), f->cache_.dXcn_dWsb(1,1), tol);
    EXPECT_NEAR(dXcn_dWsb1(2), f->cache_.dXcn_dWsb(2,1), tol);

    EXPECT_NEAR(dXcn_dWsb2(0), f->cache_.dXcn_dWsb(0,2), tol);
    EXPECT_NEAR(dXcn_dWsb2(1), f->cache_.dXcn_dWsb(1,2), tol);
    EXPECT_NEAR(dXcn_dWsb2(2), f->cache_.dXcn_dWsb(2,2), tol);
}


TEST_F(InstateJacobiansTest, Tsb) {
    Vec3 Xcn0 = ComputeXcn();

    Tsb_err(0) = delta;
    Vec3 Xcn1_0 = ComputeXcn();
    Tsb_err(0) = 0;

    Tsb_err(1) = delta;
    Vec3 Xcn1_1 = ComputeXcn();
    Tsb_err(1) = 0;

    Tsb_err(2) = delta;
    Vec3 Xcn1_2 = ComputeXcn();
    Tsb_err(2) = 0;

    Vec3 dXcn_dTsb0 = (Xcn1_0 - Xcn0) / delta;
    Vec3 dXcn_dTsb1 = (Xcn1_1 - Xcn0) / delta;
    Vec3 dXcn_dTsb2 = (Xcn1_2 - Xcn0) / delta;

    EXPECT_NEAR(dXcn_dTsb0(0), f->cache_.dXcn_dTsb(0,0), tol);
    EXPECT_NEAR(dXcn_dTsb0(1), f->cache_.dXcn_dTsb(1,0), tol);
    EXPECT_NEAR(dXcn_dTsb0(2), f->cache_.dXcn_dTsb(2,0), tol);
    
    EXPECT_NEAR(dXcn_dTsb1(0), f->cache_.dXcn_dTsb(0,1), tol);
    EXPECT_NEAR(dXcn_dTsb1(1), f->cache_.dXcn_dTsb(1,1), tol);
    EXPECT_NEAR(dXcn_dTsb1(2), f->cache_.dXcn_dTsb(2,1), tol);

    EXPECT_NEAR(dXcn_dTsb2(0), f->cache_.dXcn_dTsb(0,2), tol);
    EXPECT_NEAR(dXcn_dTsb2(1), f->cache_.dXcn_dTsb(1,2), tol);
    EXPECT_NEAR(dXcn_dTsb2(2), f->cache_.dXcn_dTsb(2,2), tol);
}


TEST_F(InstateJacobiansTest, Wbc) {
    Vec3 Xcn0 = ComputeXcn();

    Wbc_err(0) = delta;
    Vec3 Xcn1_0 = ComputeXcn();
    Wbc_err(0) = 0;

    Wbc_err(1) = delta;
    Vec3 Xcn1_1 = ComputeXcn();
    Wbc_err(1) = 0;

    Wbc_err(2) = delta;
    Vec3 Xcn1_2 = ComputeXcn();
    Wbc_err(2) = 0;

    Vec3 dXcn_dWbc0 = (Xcn1_0 - Xcn0) / delta;
    Vec3 dXcn_dWbc1 = (Xcn1_1 - Xcn0) / delta;
    Vec3 dXcn_dWbc2 = (Xcn1_2 - Xcn0) / delta;

    EXPECT_NEAR(dXcn_dWbc0(0), f->cache_.dXcn_dWbc(0,0), tol);
    EXPECT_NEAR(dXcn_dWbc0(1), f->cache_.dXcn_dWbc(1,0), tol);
    EXPECT_NEAR(dXcn_dWbc0(2), f->cache_.dXcn_dWbc(2,0), tol);
    
    EXPECT_NEAR(dXcn_dWbc1(0), f->cache_.dXcn_dWbc(0,1), tol);
    EXPECT_NEAR(dXcn_dWbc1(1), f->cache_.dXcn_dWbc(1,1), tol);
    EXPECT_NEAR(dXcn_dWbc1(2), f->cache_.dXcn_dWbc(2,1), tol);

    EXPECT_NEAR(dXcn_dWbc2(0), f->cache_.dXcn_dWbc(0,2), tol);
    EXPECT_NEAR(dXcn_dWbc2(1), f->cache_.dXcn_dWbc(1,2), tol);
    EXPECT_NEAR(dXcn_dWbc2(2), f->cache_.dXcn_dWbc(2,2), tol);
}


TEST_F(InstateJacobiansTest, Tbc) {
    Vec3 Xcn0 = ComputeXcn();

    Tbc_err(0) = delta;
    Vec3 Xcn1_0 = ComputeXcn();
    Tbc_err(0) = 0;

    Tbc_err(1) = delta;
    Vec3 Xcn1_1 = ComputeXcn();
    Tbc_err(1) = 0;

    Tbc_err(2) = delta;
    Vec3 Xcn1_2 = ComputeXcn();
    Tbc_err(2) = 0;

    Vec3 dXcn_dTbc0 = (Xcn1_0 - Xcn0) / delta;
    Vec3 dXcn_dTbc1 = (Xcn1_1 - Xcn0) / delta;
    Vec3 dXcn_dTbc2 = (Xcn1_2 - Xcn0) / delta;

    EXPECT_NEAR(dXcn_dTbc0(0), f->cache_.dXcn_dTbc(0,0), tol);
    EXPECT_NEAR(dXcn_dTbc0(1), f->cache_.dXcn_dTbc(1,0), tol);
    EXPECT_NEAR(dXcn_dTbc0(2), f->cache_.dXcn_dTbc(2,0), tol);
    
    EXPECT_NEAR(dXcn_dTbc1(0), f->cache_.dXcn_dTbc(0,1), tol);
    EXPECT_NEAR(dXcn_dTbc1(1), f->cache_.dXcn_dTbc(1,1), tol);
    EXPECT_NEAR(dXcn_dTbc1(2), f->cache_.dXcn_dTbc(2,1), tol);

    EXPECT_NEAR(dXcn_dTbc2(0), f->cache_.dXcn_dTbc(0,2), tol);
    EXPECT_NEAR(dXcn_dTbc2(1), f->cache_.dXcn_dTbc(1,2), tol);
    EXPECT_NEAR(dXcn_dTbc2(2), f->cache_.dXcn_dTbc(2,2), tol);
}





#ifdef USE_ONLINE_TEMPORAL_CALIB
TEST_F(InstateJacobiansTest, td) {
    Vec3 Xcn0 = ComputeXcn();
    td_err = delta;

    Vec3 Xcn1 = ComputeXcn();

    Vec3 dXcn_dtd = (Xcn1_0 - Xcn0) / delta;
    EXPECT_NEAR(dXcn_dtd(0), f->cache_.dXcn_dtd(0), tol);
    EXPECT_NEAR(dXcn_dtd(1), f->cache_.dXcn_dtd(1), tol);
    EXPECT_NEAR(dXcn_dtd(2), f->cache_.dXcn_dtd(2), tol);
}


#ifdef USE_ONLINE IMU_CALIB
TEST_F(InstateJacobiansTest, bg) {
    Vec3 Xcn0 = ComputeXcn();

    bg_err(0) = delta;
    Vec3 Xcn1_0 = ComputeXcn();
    bg_err(0) = 0;

    bg_err(1) = delta;
    Vec3 Xcn1_1 = ComputeXcn();
    bg_err(1) = 0;

    bg_err(2) = delta;
    Vec3 Xcn1_2 = ComputeXcn();
    bg_err(2) = 0;

    Vec3 dXcn_dbg0 = (Xcn1_0 - Xcn0) / delta;
    Vec3 dXcn_dbg1 = (Xcn1_1 - Xcn0) / delta;
    Vec3 dXcn_dbg2 = (Xcn1_2 - Xcn0) / delta;

    EXPECT_NEAR(dXcn_dbg0(0), f->cache_.dXcn_dbg(0,0), tol);
    EXPECT_NEAR(dXcn_dbg0(1), f->cache_.dXcn_dbg(1,0), tol);
    EXPECT_NEAR(dXcn_dbg0(2), f->cache_.dXcn_dbg(2,0), tol);
    
    EXPECT_NEAR(dXcn_dbg1(0), f->cache_.dXcn_dbg(0,1), tol);
    EXPECT_NEAR(dXcn_dbg1(1), f->cache_.dXcn_dbg(1,1), tol);
    EXPECT_NEAR(dXcn_dbg1(2), f->cache_.dXcn_dbg(2,1), tol);

    EXPECT_NEAR(dXcn_dbg2(0), f->cache_.dXcn_dbg(0,2), tol);
    EXPECT_NEAR(dXcn_dbg2(1), f->cache_.dXcn_dbg(1,2), tol);
    EXPECT_NEAR(dXcn_dbg2(2), f->cache_.dXcn_dbg(2,2), tol);
}
#endif
#endif