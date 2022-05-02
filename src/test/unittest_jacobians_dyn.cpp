#include <gtest/gtest.h>
//#include <unsupported/Eigen/MatrixFunctions>

#define private public

#include "estimator.h"

#include "unittest_helpers.h"


using namespace xivo;


class DynamicsJacobiansTest : public ::testing::Test {
  protected:
    void SetUp() override {

        // Create Estimator with tumvi benchmark parameters
        auto cfg = LoadJson("cfg/tumvi.json");
        est = CreateSystem(LoadJson(cfg["estimator_cfg"].asString()));
        delta = 1e-6;
        tol = 1e-5;

    }

    void SetNonZeroState () {
        est->X_.Rsb = SO3::exp({1.1, 2.2, 3.3});
        est->X_.Tsb = {3, -5, 7};
        est->X_.Vsb = {0.2, 0.1, -0.4};
        est->X_.ba = {1e-3, 2.5e-3, -5.0e-4};
        est->X_.bg = {-3e-4, -1.1e-4, 2e-4};
        est->X_.Rbc = SO3::exp({1.87, 1.98, -0.0048}); // phab number
        est->X_.Tbc = {-0.025, 0.025, -0.040};
        // from end of seq data9_workbench
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

        // something small
        imu_bias_noise_input << 0.001, 0.001, -0.001, 0.01, -0.01, 0.01;

        // {0}
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

    void NonlinearDynamicsFcn(State::Tangent &xdot,
                              IMUState::Tangent &imuderiv) {
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

        imuderiv.setZero();
    }

    number_t CalcNumericalJac(int i,
                              State::Tangent x_deriv0,
                              State::Tangent x_deriv1,
                              IMUState::Tangent imu_deriv0,
                              IMUState::Tangent imu_deriv1) {
        number_t num_jac;
#ifdef USE_ONLINE_IMU_CALIB
        if (i < Index::Cg) {
            num_jac = (x_deriv1(i) - x_deriv0(i)) / delta;
        } else {
            num_jac = (imu_deriv1(i-Index::Cg) - imu_deriv0(i-Index::Cg)) / delta;
        }
#else
        num_jac = (x_deriv1(i) - x_deriv0(i)) / delta;
#endif
        return num_jac;
    }

    void RunTests(std::string errmsg_start) {
        // Compute Analytical Jacobian
        est->ComputeMotionJacobianAt(est->X_, imu_input);

        // Variables to hold perturbed derivatives and backup state
        State::Tangent x_deriv1;
        IMUState::Tangent imu_deriv1;
        State::Tangent x_deriv2;
        IMUState::Tangent imu_deriv2;
        State X_backup = est->X_;
        IMUState imu_backup = est->imu_.X_;

        number_t num_jac;

        // Compute numerical Jacobians in F_ one at a time
        // We are numerically approximating the derivative of element i with
        // respect to element j
        // Note that since kMotionSize includes td, Cg, Ca, we will only end up
        // testing those derivatives if they are part of the state
        for (int i=0; i<kMotionSize; i++) {
            for (int j=0; j<kMotionSize; j++) {

                // make perturbation in element j of the total state vector
                PerturbElement(j, -delta/2, est->X_, est->imu_.X_);
                NonlinearDynamicsFcn(x_deriv1, imu_deriv1);
                est->X_ = X_backup;
                est->imu_.X_ = imu_backup;

                PerturbElement(j, delta/2, est->X_, est->imu_.X_);
                NonlinearDynamicsFcn(x_deriv2, imu_deriv2);
                est->X_ = X_backup;
                est->imu_.X_ = imu_backup;

                // Compute numerical jacobian of state i dynamics w.r.t. state j
                num_jac = CalcNumericalJac(i, x_deriv1, x_deriv2, imu_deriv1,
                                           imu_deriv2);
                EXPECT_NEAR(num_jac, est->F_.coeff(i,j), tol) <<
                    errmsg_start <<
                    "State jacobian error at state " << i << ", state " << j;

                // Put state back where it was
                est->X_ = X_backup;
                est->imu_.X_ = imu_backup;
            }
        }

        // Compute numerical Jacobians in G_ w.r.t. measurement noise
        Vec6 imu_noise_input_backup = imu_noise_input;

        for (int i=0; i<kMotionSize; i++) {
            for (int j=0; j<6; j++) {
                imu_noise_input(j) -= delta/2;
                NonlinearDynamicsFcn(x_deriv1, imu_deriv1);
                imu_noise_input = imu_noise_input_backup;

                imu_noise_input(j) += delta/2;
                NonlinearDynamicsFcn(x_deriv2, imu_deriv2);
                imu_noise_input = imu_noise_input_backup;

                num_jac = CalcNumericalJac(i, x_deriv1, x_deriv2, imu_deriv1,
                                           imu_deriv2);
                EXPECT_NEAR(num_jac, est->G_.coeff(i,j), tol) <<
                    errmsg_start <<
                    "Input jacobian error at state " << i << ", input " << j;
            }
        }

        // Compute numerical Jacobians in G_ w.r.t. bias noise
        Vec6 imu_bias_noise_input_backup = imu_bias_noise_input;
        for (int i=0; i<kMotionSize; i++) {
            for (int j=0; j<6; j++) {
                imu_bias_noise_input(j) -= delta/2;
                NonlinearDynamicsFcn(x_deriv1, imu_deriv1);
                imu_bias_noise_input = imu_bias_noise_input_backup;

                imu_bias_noise_input(j) += delta/2;
                NonlinearDynamicsFcn(x_deriv2, imu_deriv2);
                imu_bias_noise_input = imu_bias_noise_input_backup;

                num_jac = CalcNumericalJac(i, x_deriv1, x_deriv2, imu_deriv1,
                                           imu_deriv2);
                EXPECT_NEAR(num_jac, est->G_.coeff(i,6+j), tol) <<
                    errmsg_start <<
                    "Input jacobian error at state " << i << ", input " << j + 6;

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
