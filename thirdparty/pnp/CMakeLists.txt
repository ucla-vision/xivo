cmake_minimum_required(VERSION 3.16)
project(lambdatwist-pnp CXX) # defines ${CMAKE_PROJECT_NAME}(top level) PROJECT_NAME(most recent)

message("================================================================================")
message("Builing pnp")
message("Recommended optimization flags for release:  -O3 -march=native -mtune=native -DNDEBUG")




# Build configuration
macro(BuildConfig)

    set(default_build_type "Release")
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
        set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
        # Set the possible values of build type for cmake-gui
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "RelWithDebInfo")
    endif()
endmacro()
BuildConfig()



# note, ceres 1.4 works just fine too, but there is a bug in the cmakelists of ceres 1.4 which
# can cause eigen3 lookup failures.
message(" Ceres 1.4 works just fine too, but there are cmake errors in them which may cause eigen3 to not be found. In that case add it manually, but be careful, as if ceres is built with one version of eigen, and then used in combination with another, this will cause segfaults inside ceres due to the idiotic ideas around aligned allocators,(just dont use them is the solution ceres and eigen does not appear to get)")
find_package(Ceres 2 REQUIRED )

add_library(libpnp utils/mlibtime.cpp pnp_ransac.cpp p4p.cpp)
target_link_libraries(libpnp PRIVATE ceres)



option(PNP_RANDOM_SEED_FROM_TIME "generate a seed from time, default" OFF)
if(PNP_RANDOM_SEED_FROM_TIME)
    target_compile_definitions(libpnp PRIVATE "RANDOM_SEED_FROM_TIME")
else()
    target_compile_definitions(libpnp PRIVATE "RANDOM_SEED_VALUE=0")
endif()

# the compiler does not care about having multiple equivalent but distinct includes, but many ides do...
get_filename_component(PNP_PATH ${CMAKE_CURRENT_SOURCE_DIR} REALPATH)
target_include_directories(libpnp PUBLIC ${PNP_PATH})
target_compile_features(libpnp PUBLIC cxx_std_11)
set_target_properties(libpnp PROPERTIES POSITION_INDEPENDENT_CODE ON)


option(WITH_PYBIND11 "with pybind11 bindings" OFF)
if(WITH_PYBIND11)
    message("requires that submodules are inited! git submodule init && git submodule update")
    add_subdirectory(pybind11)
    pybind11_add_module(pnp_python_binding pnp_python_binding.cpp)
    target_link_libraries(pnp_python_binding PUBLIC libpnp )
    message("OBS! Pybind11 require that you specify python exectuable to build for, e.g. cmake .. -DPYTHON_EXECUTABLE=python3")
    # OBS! Pybind11 require that you specify python exectuable to build for, e.g. cmake .. -DPYTHON_EXECUTABLE=python3
endif()

set(BUILD_PNP_MAIN_DEFAULT OFF)
message("${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR}")
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message("${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR}")
    set(BUILD_PNP_MAIN_DEFAULT ON)
endif()
option(BUILD_PNP_MAIN "build pnp main" ${BUILD_PNP_MAIN_DEFAULT})
if(${BUILD_PNP_MAIN})
    add_executable(main test_pnp.cpp)
    target_link_libraries(main libpnp)
endif()

message("================================================================================")
