cmake_minimum_required(VERSION 3.5)
project(common)

add_library(common STATIC utils.cpp)

# option(BUILD_COMMON_TESTS True)
# if (${BUILD_COMMON_TESTS})
#   add_executable(test_rodrigues test/test_rodrigues.cpp)
#   target_link_libraries(test_rodrigues gtest gtest_main)
# endif(${BUILD_COMMON_TESTS})
