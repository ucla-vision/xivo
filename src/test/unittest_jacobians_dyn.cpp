#include <gtest/gtest.h>
//#include <unsupported/Eigen/MatrixFunctions>

#define private public

#include "rodrigues.h"
#include "estimator.h"

#include "unittest_helpers.h"


using namespace xivo;


class DynamicsJacobiansTest : public ::testing::Test {
  protected:
    void SetUp() override {

        // Create Estimator with tumvi benchmark parameters
        auto cfg = LoadJson("cfg/tumvi.json");
        est = CreateSystem(LoadJson(cfg["estimator_cfg"].asString()));
        delta = 1e-8;
        tol = 1e-5;

    }

    void SetNonZeroState () {
        //est->X_.Rsb = SO3::exp({1.1, 2.2, 3.3});
        est->X_.Rsb = SO3::exp({-0.579252,   -1.1585,  -1.73776});
        est->X_.Tsb = {3, -5, 7};
        est->X_.Vsb = {0.2, 0.1, -0.4};
        est->X_.ba = {1e-3, 2.5e-3, -5.0e-4};
        est->X_.bg = {-3e-4, -1.1e-4, 2e-4};
        est->X_.Rbc = SO3::exp({1.87, 1.98, -0.0048}); // phab number
        est->X_.Tbc = {-0.025, 0.025, -0.040};
        est->X_.Rg = SO3::exp({0.0123478, -1.301, 0});

#ifdef USE_ONLINE_TEMPORAL_CALIB
        est->X_.td = -0.001;
#endif

        // Values from phab phone
        Mat3 Ta, Ka;
        Ta << 1, 0.00533542, 0.00268388,
              0,          1, -0.0107169,
              0,          0,          1;
        Ka << 0.997708, 0.0, 0.0,
              0.0, 0.997608, 0.0,
              0.0, 0.0, 0.987496;
        Mat3 Tg, Kg;
        Tg << 1,   -0.0115696, -0.000669818,
              -0.00974687,  1,  -0.00995586,
               0.00887142,    0.0119916,  1;
        Kg << 1.03991, 0.0, 0.0,
              0.0, 1.04305, 0.0,
              0.0, 0.0, 1.05785;
        
        est->imu_.X_.Ca = Ta * Ka;
        est->imu_.X_.Cg = Tg * Kg;

        // something out of the phab IMU
        imu_input << -0.00079345703125, -0.000746657228125, -0.0017173580942,
                     -9.57653808594, 0.134033203125, 1.72415161133;

        imu_noise_input.setZero();
        imu_bias_noise_input.setZero();

        est->g_ = { 0.0, 0.0, -9.796 };
    }

    void SetIdentityState() {
        est->X_.Rsb = SO3();
        est->X_.Tsb.setZero();
        est->X_.Vsb.setZero();
        est->X_.ba.setZero();
        est->X_.bg.setZero();
        est->X_.Rbc = SO3();
        est->X_.Tbc.setZero();
        est->X_.Rg = SO3();

#ifdef USE_ONLINE_TEMPORAL_CALIB
        est->X_.td = 0;
#endif

        est->imu_.X_.Ca.setIdentity();
        est->imu_.X_.Cg.setIdentity();

        // no imu input
        imu_input.setZero();
        imu_bias_noise_input.setZero();
        imu_noise_input.setZero();

        // default gravity
        est->g_ = {0, 0, -9.8};
    }

    void PerturbElement(int j, number_t delta, State &X, IMUState &imu) {
#ifdef USE_ONLINE_IMU_CALIB
        State::Tangent X_tangent;
        IMUState::Tangent IMU_tangent;
        if ((j >= Index::Cg) && (j < Index::Ca)) {
            IMU_tangent(6 + (j - Index::Cg)) += delta;
        } else if (j >= Index::Ca) {
            IMU_tangent(j - Index::Ca) += delta;
        } else {
            X_tangent(j) += delta;
        }
        X += X_tangent;
        imu += IMU_tangent;
#else
        State::Tangent X_tangent;
        X_tangent(j) += delta;
        X += X_tangent;
#endif
    }

    void NonlinearDynamicsFcn(State::Tangent &xdot) {
        Vec3 gyro_input = imu_input.head<3>();
        Vec3 accel_input = imu_input.tail<3>();

        Vec3 gyro_calib = est->Cg() * gyro_input - est->bg() -
            imu_noise_input.head<3>();
        Vec3 accel_calib = est->Ca() * accel_input - est->ba() -
            imu_noise_input.tail<3>();

        SE3 gsb = est->gsb();
        Mat3 Rsb = gsb.R().matrix();
        Mat3 Rsg = est->Rg().matrix();

        // Calculate jacobian of Wsbdot
        Mat3 Rsbdot = Rsb * hat(gyro_calib);
        Vec9 Rsbdot_flat = Eigen::Map<Vec9> (Rsbdot.transpose().data());
        Mat39 dWsb_dRsb;
        Vec3 Wsb = invrodrigues(Rsb, &dWsb_dRsb);
        for (int i = 0; i < 3; i++) {
            xdot(Index::Wsb + i) = dWsb_dRsb.row(i) * Rsbdot_flat;
        }

        xdot.segment<3>(Index::Tsb) = est->Vsb();
        xdot.segment<3>(Index::Vsb) = Rsb * accel_calib + Rsg * est->g_;
        xdot.segment<3>(Index::bg) = imu_bias_noise_input.head<3>();
        xdot.segment<3>(Index::ba) = imu_bias_noise_input.tail<3>();
        xdot.segment<3>(Index::Wbc).setZero();
        xdot.segment<3>(Index::Tbc).setZero();
        xdot.segment<2>(Index::Wg).setZero();

#ifdef USE_ONLINE_TEMPORAL_CALIB
        xdot(Index::td) = 0.0;
#endif
    }

    void RunTests(std::string errmsg_start) {
        // Compute Analytical Jacobian
        est->ComputeMotionJacobianAt(est->X_, imu_input);

        // Variables to hold perturbed derivatives and backup state
        State::Tangent x_deriv0;
        State::Tangent x_deriv1;
        State X_backup = est->X_;
        IMUState imu_backup = est->imu_.X_;

        NonlinearDynamicsFcn(x_deriv0);

        // Compute numerical Jacobians in F_ one at a time
        // We are numerically approximating the derivative of element i with
        // respect to element j
        // Note that since kMotionSize includes td, Cg, Ca, we will only end up
        // testing those derivatives if they are part of the state
        for (int j=0; j<kMotionSize; j++) {
            PerturbElement(j, delta, est->X_, est->imu_.X_);
            NonlinearDynamicsFcn(x_deriv1); 
            est->X_ = X_backup;
            est->imu_.X_ = imu_backup;

            Eigen::Matrix<number_t, kMotionSize, 1> num_jac =
                (x_deriv1 - x_deriv0) / delta;
            for (int i=0; i<kMotionSize; i++) {
                // Skip pathological cases that always fail because of numerical
                // issues that arise when trying to make small changes to
                // Rsb through a matrix multiplication with Wsb. These cases,
                // however, are separately coded in a less pathological way in
                // other unit tests that do pass. A printout of the analytical
                // jacobians computed here match the analytical and numeric
                // jacobians in those other test cases.
                // Pathological test cases:
                // - dV_dWsb
                // - dV_dWsg
                if ((Index::V <= i) && (i < Index::V+3)) {
                    if ((Index::Wsb <= j) && (j < Index::Wsb+3)) {
                        continue;
                    }
                    if ((Index::Wg <= j) && (j < Index::Wg+2)) {
                        continue;
                    }
                }

                EXPECT_NEAR(num_jac(i), est->F_.coeff(i,j), tol) <<
                    errmsg_start <<
                    "State jacobian error at state " << i << ", state " << j
                    << std::endl;
            }
        }

        // print analytical dV_dw and dV_dwg. These match both the analytical
        // and numerical values in unit tests `DynamicsJacobiansTest.dV_dWsb`
        // and `DynamicsJacobiansTest.dV_dWsg` that do pass because the small
        // increment to `Rsb` is applied differently when computing the
        // numerical derivative.
        /*
        Mat3 dV_dw;
        Mat3 dV_dwg;
        dV_dw.setZero();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                dV_dw(i,j) = est->F_.coeff(Index::V+i, Index::Wsb+j);
            }
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                dV_dwg(i,j) = est->F_.coeff(Index::V+i, Index::Wg+j);
            }
        }
        std::cout << "analytical jac dV_dw" << std::endl << dV_dw.format(Eigen::FullPrecision) << std::endl;
        std::cout << "analytical jac dV_dwsg" << std::endl << dV_dwg.format(Eigen::FullPrecision) << std::endl;
        */

        // Compute numerical Jacobians in G_ w.r.t. measurement noise
        for (int j=0; j<6; j++) {
            imu_noise_input(j) += delta;
            NonlinearDynamicsFcn(x_deriv1);
            imu_noise_input(j) -= delta;

            Eigen::Matrix<number_t, kMotionSize, 1> num_jac =
                (x_deriv1 - x_deriv0) / delta;
            for (int i=0; i<kMotionSize; i++) {
                EXPECT_NEAR(num_jac(i), est->G_.coeff(i,j), tol) <<
                    errmsg_start <<
                    "Input jacobian error at state " << i << ", input " << j <<
                    std::endl;
            }
        }

        // Compute numerical Jacobians in G_ w.r.t. bias noise
        for (int j=0; j<6; j++) {
            imu_bias_noise_input(j) += delta;
            NonlinearDynamicsFcn(x_deriv1);
            imu_bias_noise_input(j) -= delta;
            Eigen::Matrix<number_t, kMotionSize, 1> num_jac =
                (x_deriv1 - x_deriv0) / delta;

            for (int i=0; i<kMotionSize; i++) {
                EXPECT_NEAR(num_jac(i), est->G_.coeff(i,6+j), tol) <<
                    errmsg_start <<
                    "Input jacobian error at state " << i << ", input " << j + 6
                    << std::endl;
            }
        }
    }

    // Estimator Object
    EstimatorPtr est;
    Vec6 imu_input;
    Vec6 imu_noise_input;
    Vec6 imu_bias_noise_input;

    number_t tol;   // numerical tolerance for checks
    number_t delta; // finite difference
};


TEST_F(DynamicsJacobiansTest, Zero) {
    SetIdentityState();
    RunTests("Identity State Jacobians: ");
}


TEST_F(DynamicsJacobiansTest, NonZero) {
    SetNonZeroState();
    RunTests("Nonzero State Jacobians: ");
}

TEST_F(DynamicsJacobiansTest, dV_dWsb) {
    SetNonZeroState();
    est->ComputeMotionJacobianAt(est->X_, imu_input);

    Vec3 Wsb = est->X_.Rsb.log();
    Vec3 accel_calib = est->imu_.Ca() * imu_input.tail<3>() - est->ba();
    Mat3 jac;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            jac(i,j) = est->F_.coeffRef(Index::V+i, Index::Wsb+j);
        }
    }

    Vec3 y0 = est->X_.Rsb.matrix() * accel_calib;
    Mat3 num_jac;

    for (int j = 0; j < 3; j++) {
        Vec3 wp(Wsb);
        wp(j) += delta;
        Vec3 y1 = rodrigues(wp) * accel_calib;

        num_jac.col(j) = (y1 - y0) / delta;
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(jac(i,j), num_jac(i,j), 1e-5) <<
            "dV_dWsb test error at i=" << i << " and j=" << j;
        }
    }
}

TEST_F(DynamicsJacobiansTest, dV_dWsg) {
    SetNonZeroState();
    est->ComputeMotionJacobianAt(est->X_, imu_input);

    Vec3 Wsg = est->X_.Rg.log();
    Vec3 y0 = est->X_.Rg.matrix() * est->g_;
    Mat3 jac; // analytical Jacobians
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            jac(i,j) = est->F_.coeffRef(Index::V+i, Index::Wg+j);
        }
    }

    Mat3 num_jac;
    for (int j = 0; j < 2; j++) {
        Vec3 wp(Wsg);
        wp(j) += delta;
        Vec3 y1 = rodrigues(wp) * est->g_;
        num_jac.col(j) = (y1 - y0) / delta;
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            EXPECT_NEAR(jac(i,j), num_jac(i,j), 1e-5) <<
            "dV_dWsg test error at i=" << i << " and j=" << j;
        }
    }
}


TEST_F(DynamicsJacobiansTest, dW_dW) {
    SetNonZeroState();
    est->ComputeMotionJacobianAt(est->X_, imu_input);

    Mat3 jac;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            jac(i,j) = est->F_.coeffRef(Index::Wsb+i, Index::Wsb+j);
        }
    }

    // First numeric value
    Mat3 Rsb = est->X_.Rsb.matrix();
    Vec3 Wsb = est->X_.Rsb.log();
    Mat3 Cg = est->imu_.Cg();
    Vec3 gyro = imu_input.head<3>();
    Vec3 bg = est->X_.bg;
    Vec3 gyro_calib = Cg * gyro - bg;
    Vec3 y0 = Rsb * gyro_calib;

    // Perturbed numeric values and numeric jacobian
    Mat3 num_jac;
    for (int j = 0; j < 3; j++) {
        Vec3 wp(Wsb);
        wp(j) += delta;
        Vec3 y1 = rodrigues(wp) * gyro_calib;
        num_jac.col(j) = (y1 - y0) / delta;
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(jac(i,j), num_jac(i,j), 1e-5) <<
            "dW_dWsb test error at i=" << i << " and j=" << j;
        }
    }

}