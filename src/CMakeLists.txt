cmake_minimum_required(VERSION 3.5)

project(estimator)

# # overwrite binary output directory
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# if set, use inverse-depth parametrization
# add_definitions(-DUSE_INVDEPTH)

# if set, perform online temporal calibration
add_definitions(-DUSE_ONLINE_IMU_CALIB)
add_definitions(-DUSE_ONLINE_TEMPORAL_CALIB)
add_definitions(-DUSE_ONLINE_CAMERA_CALIB)

# if set, approximate the initial correlation between feature state and
# group state with Hessian from depth refinement optimization.
# WARNING: this feature not work yet.
# add_definitions(-DAPPROXIMATE_INIT_COVARIANCE)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/common)

link_directories(
  ${CMAKE_SOURCE_DIR}/lib)

list(APPEND deps
        opencv_highgui
        opencv_features2d
        opencv_core
        opencv_video
        opencv_imgproc
        opencv_imgcodecs
        opencv_xfeatures2d
        glog
        pangolin
        pthread
        jsoncpp
        GL
        GLEW
        common
        )

if (USE_GPERFTOOLS)
  list(APPEND deps profiler)
endif (USE_GPERFTOOLS)

add_library(xapp STATIC
        estimator_process.cpp
        loader.cpp
        geometry.cpp
        metrics.cpp
        publisher.cpp
        viewer.cpp)
target_link_libraries(xapp ${deps})

add_library(xest STATIC
        factory.cpp
        estimator.cpp
        princedormand.cpp
        rk4.cpp
        visualize.cpp
        tracker.cpp
        manager.cpp
        update.cpp
        graph.cpp
        feature.cpp
        oos.cpp
        group.cpp
        helpers.cpp
        options.cpp
        param.cpp
        mm.cpp
        camera_manager.cpp
        imu.cpp)
target_link_libraries(xest INTERFACE ${deps} "-Wl,-lstdc++fs")

set(libxivo xest xapp)

if (BUILD_G2O)
  add_library(xopt STATIC
    optimizer.cpp
    optimizer_adapters.cpp
  )
  target_link_libraries(xopt xest ${deps})
  list(APPEND libxivo xopt)
endif(BUILD_G2O)

add_executable(vio app/vio.cpp)
target_link_libraries(vio ${libxivo} gflags::gflags)

################################################################################
# TOOLING
################################################################################
# add_executable(eval app/evaluate.cpp)
# target_link_libraries(eval estimator application gflags::gflags)

################################################################################
# TESTS
################################################################################
add_executable(unitTests_NumericalAlgorithms 
               test/unittest_givens.cpp
               test/unittest_helpers.cpp)
target_link_libraries(unitTests_NumericalAlgorithms ${libxivo} ${deps} gtest gtest_main)
add_test(NAME NumericalAlgorithms COMMAND unitTests_NumericalAlgorithms)

add_executable(unitTests_Jacobians
               test/unittest_jacobians_instate.cpp
               test/unittest_jacobians_oos.cpp
               test/unittest_helpers.cpp)
target_link_libraries(unitTests_Jacobians ${libxivo} ${deps} gtest gtest_main)
add_test(NAME Jacobians COMMAND unitTests_Jacobians)

add_executable(unitTests_Rodrigues
               test/unittest_rodrigues.cpp)
target_link_libraries(unitTests_Rodrigues ${libxivo} ${deps} gtest gtest_main)
add_test(NAME Rodrigues COMMAND unitTests_Rodrigues)

add_executable(unitTests_pinhole
               test/unittest_camera_pinhole.cpp)
target_link_libraries(unitTests_pinhole ${libxivo} ${deps} gtest gtest_main)
add_test(NAME CamerasPinhole COMMAND unitTests_pinhole)

add_executable(unitTests_radtan
               test/unittest_camera_radtan.cpp)
target_link_libraries(unitTests_radtan ${libxivo} ${deps} gtest gtest_main)
add_test(NAME CamerasRadtan COMMAND unitTests_radtan)

add_executable(unitTests_equi
               test/unittest_camera_equi.cpp)
target_link_libraries(unitTests_equi ${libxivo} ${deps} gtest gtest_main)
add_test(NAME CamerasEqui COMMAND unitTests_equi)

add_executable(unitTests_atan
               test/unittest_camera_atan.cpp)
target_link_libraries(unitTests_atan ${libxivo} ${deps} gtest gtest_main)
add_test(NAME CamerasAtan COMMAND unitTests_atan)

if (BUILD_G2O)
  message(INFO ${libxivo})
  add_executable(test_optimizer test/test_optimizer.cpp)
  target_link_libraries(test_optimizer ${libxivo})
endif(BUILD_G2O)

# add_executable(test_estimator test/test_estimator.cpp )
# target_link_libraries(test_estimator estimator gtest gtest_main)
#
# add_executable(test_simulator test/test_simulator.cpp )
# target_link_libraries(test_simulator estimator gtest gtest_main)

# add_executable(test_tumvi test/test_tumvi.cpp )
# target_link_libraries(test_tumvi estimator gtest gtest_main)

# add_executable(test_viewer test/test_viewer.cpp)
# target_link_libraries(test_viewer estimator gtest gtest_main)

# add_executable(test_tracker test/test_tracker.cpp )
# target_link_libraries(test_tracker estimator gtest gtest_main)
#
# add_executable(test_matcher test/test_matcher.cpp )
# target_link_libraries(test_matcher estimator gtest gtest_main)
