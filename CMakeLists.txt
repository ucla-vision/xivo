cmake_minimum_required(VERSION 3.5)
project(feh)

# Set operating system variables
if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(IS_LINUX TRUE)
endif ()
if (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  set(IS_MAC TRUE)
endif ()
message(STATUS "IS_LINUX ${IS_LINUX}")
message(STATUS "IS_MAC ${IS_MAC}")

option(BUILD_G2O "build with g2o support" OFF)
option(USE_GPERFTOOLS "use gperf for performance profiling" OFF)

if (USE_GPERFTOOLS)
  add_definitions(-DUSE_GPERFTOOLS)
endif (USE_GPERFTOOLS)

if (BUILD_G2O)
  add_definitions(-DUSE_G2O)
endif (BUILD_G2O)

set(CMAKE_CXX_STANDARD 17)
# TODO (xiaohfei): We should turn on -Wall and -Werror to eliminate all the warnings.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -Wno-narrowing -Wno-register -fPIC -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")

# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_BUILD_TYPE "Release")
#add_definitions(-DNDEBUG)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# disable logging whose severity level is below the given integer
add_definitions(-DGOOGLE_STRIP_LOG=1)

# add_definitions(-DEIGEN_DEFAULT_TO_ROW_MAJOR)
add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_ZERO)

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/thirdparty)

# Find and add OpenCV to libraries
find_package(OpenCV REQUIRED)
find_package(Python3 REQUIRED Interpreter Development)
find_package(Pangolin REQUIRED)
find_package(jsoncpp REQUIRED)
# Google stuff
find_package(gflags REQUIRED)
find_package(GTest REQUIRED NAMES GTest gtest googletest GoogleTest)
find_package(glog REQUIRED)
find_package(Ceres REQUIRED)
# Find and add DBoW2 
find_package(DBoW2 REQUIRED)
# Create an interface target for DBoW2 (since DBoW2 does not export headers & libraries as a target)
add_library(DBoW2 INTERFACE)
target_include_directories(DBoW2 INTERFACE ${DBoW2_INCLUDE_DIR})
target_link_libraries(DBoW2 INTERFACE ${DBoW2_LIBS})

# TODO (xiaohfei): this is not working since gperftools does not properly generate cmake config files.
# Need to manually add gperftools if needed.
if (USE_GPERFTOOLS)
  find_package(gperftools REQUIRED)
endif()

# The dependency PNP does have proper installation target, need to manually specify paths.
add_library(pnp INTERFACE)
target_include_directories(pnp INTERFACE ${CMAKE_SOURCE_DIR}/thirdparty/pnp)
target_link_libraries(pnp INTERFACE ${CMAKE_SOURCE_DIR}/thirdparty/pnp/build/liblibpnp.a)

list(APPEND DEPS
  # opencv
  opencv_highgui
  opencv_features2d
  opencv_core
  opencv_video
  opencv_imgproc
  opencv_imgcodecs
  opencv_xfeatures2d
  # google stuff
  glog::glog
  GTest::gtest
  GTest::gtest_main
  GTest::gmock
  GTest::gmock_main
  gflags_static
  # graphics
  GLEW
  pangolin
  # solver
  Ceres::ceres
  pnp
  cxsparse
  cholmod
  lapack
  blas
  # utils
  jsoncpp_lib
  # Use jsoncpp_static if you are building modules of xivo as static libraryies
  # jsoncpp_static
  DBoW2
  # system
  pthread
)
if (IS_LINUX)
  list(APPEND DEPS GL)
endif (IS_LINUX)

link_libraries(${DEPS})

# Manually add link directories (ideally we dont have to use this)
link_directories(
  /usr/local/lib
  /usr/lib/x86_64-linux-gnu
)

include_directories(
  ${CMAKE_SOURCE_DIR}/common
  ${CMAKE_SOURCE_DIR}/src
  # Manually add include dirs, ideally we don't need this
  # ${JSONCPP_INCLUDE_DIRS}
  # ${OpenCV_INCLUDE_DIRS}
  # ${Python3_INCLUDE_DIRS}
  /usr/include/suitesparse
  /usr/include
)



enable_testing()
# add_subdirectory(thirdparty/abseil-cpp)
if (BUILD_G2O)
  link_directories(${CMAKE_SOURCE_DIR}/thirdparty/g2o/release/lib)
  include_directories(${CMAKE_SOURCE_DIR}/thirdparty/g2o/release/include)
  list(APPEND deps
    g2o_core
    g2o_solver_dense
    g2o_solver_cholmod
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_types_slam3d
    g2o_types_sba
    g2o_stuff
    cholmod
    cxsparse
    )
endif(BUILD_G2O)

# feh
add_subdirectory(common)
add_subdirectory(src)

########################################
# PYTHON BINDING
########################################
# NOTE: to build with a specific python version
# cmake -DPYTHON_EXECUTABLE=path/to/python ..
# By default, the python binding generated is only compatible with
# your default python interpreter, which you can check by typing
# "which python" in your terminal.
# If you see an error saying "Python.h: No such file or directory", you probably
# need to "sudo apt-get install python3-dev" assuming you are binding to python3.
set(PYBIND11_CPP_STANDARD -std=c++17)
add_subdirectory(thirdparty/pybind11)
pybind11_add_module(pyxivo MODULE pybind11/pyxivo.cpp)
set(libxivo common xest xapp)
if (BUILD_G2O)
  list(APPEND libxivo xopt)
endif(BUILD_G2O)
target_link_libraries(pyxivo PRIVATE ${libxivo})
