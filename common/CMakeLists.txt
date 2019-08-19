cmake_minimum_required(VERSION 3.5)
project(common)

# # uncomment to overwrite target destinations
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(common STATIC utils.cpp)
target_link_libraries(common absl::str_format pthread)

# add_executable(example_process example_process.cpp)
# target_link_libraries(example_process glog pthread)
#
# option(BUILD_COMMON_TESTS True)
# if (${BUILD_COMMON_TESTS})
#   add_executable(test_rodrigues test/test_rodrigues.cpp)
#   target_link_libraries(test_rodrigues gtest gtest_main)
#   # add_executable(test_se3 test_se3.cpp)
# endif(${BUILD_COMMON_TESTS})
