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

# disable logging whose severity level is below the given integer
add_definitions(-DGOOGLE_STRIP_LOG=1)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/common)

link_directories(
  ${CMAKE_SOURCE_DIR}/lib)

set(deps
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

add_library(application SHARED
        estimator_process.cpp
        tumvi.cpp
        geometry.cpp
        metrics.cpp
        publisher.cpp
        viewer.cpp)
target_link_libraries(application ${deps})

add_library(estimator SHARED
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
        core.cpp
        helpers.cpp
        options.cpp
        param.cpp
        mm.cpp
        camera_manager.cpp
        imu.cpp)
target_link_libraries(estimator ${deps}) # profiler)

add_executable(vio app/vio.cpp)
target_link_libraries(vio estimator application gflags::gflags)

add_executable(st_vio app/singlethread_vio.cpp)
target_link_libraries(st_vio estimator application gflags::gflags)

################################################################################
# TOOLING
################################################################################
# add_executable(eval app/evaluate.cpp)
# target_link_libraries(eval estimator application gflags::gflags)

################################################################################
# TESTS
################################################################################
# add_executable(test_givens test/test_givens.cpp )
# target_link_libraries(test_givens estimator)
#
# add_executable(test_qr test/test_qr.cpp )
# target_link_libraries(test_qr estimator)

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
