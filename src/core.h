// Core data structures include: 
// 1) timestamp types
// 2) representation of the nominal state (State)
// 3) layout of the error state (Index and other offsets)
// 4) status of components, etc.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <chrono>
#include <exception>
#include <iostream>
#include <list>
#include <memory>

#include "alias.h"
#include "camera_manager.h"
#include "helpers.h"
#include "rodrigues.h"
#include "se3.h"
#include "utils.h"

namespace feh {
////////////////////////////////////////
// TYPES FOR TIME
////////////////////////////////////////
using nanoseconds = std::chrono::nanoseconds;  // uint64_t
using seconds = std::chrono::duration<double>; // double
using timestamp_t = nanoseconds;

////////////////////////////////////////
// STATE DIMENSION
////////////////////////////////////////
enum Index : int {
  W = 0, // Wsb, rotation
  Wsb = 0,
  T = 3, // Tsb, translation
  Tsb = 3,
  V = 6, // vsb, velocity
  Vsb = 6,
  bg = 9,   // omega bias
  ba = 12,  // alpha bias
  Wbc = 15, // alignment rotation
  Tbc = 18, // alignment translation
  Wg = 21,  // gravity
#ifdef USE_ONLINE_TEMPORAL_CALIB
  td = 24, // temporal offset
#endif

#ifdef USE_ONLINE_IMU_CALIB

#ifdef USE_ONLINE_TEMPORAL_CALIB
  Cg = 25, // gyro calibration, 9 numbers
#else
  Cg = 24, // gyro calibration, 9 numbers
#endif
  Ca = Cg + 9, // accel calibration, 6 numbers
  End = Ca + 6,

#else

#ifdef USE_ONLINE_TEMPORAL_CALIB
  End = td + 1,
#else
  End = Wg + 3
#endif

#endif

};

constexpr int kMotionSize = Index::End;

constexpr int kCameraBegin = kMotionSize;

#ifdef USE_ONLINE_CAMERA_CALIB
constexpr int kMaxCameraIntrinsics =
    9; // maximal possible number of intrinsic parameters
#else
constexpr int kMaxCameraIntrinsics =
    0; // maximal possible number of intrinsic parameters
#endif

constexpr int kGroupSize = 6;
constexpr int kFeatureSize = 3;

constexpr int kMaxFeature = 30;
constexpr int kMaxGroup = kMaxFeature;  // worst case, each feature is attached to a different reference group

constexpr int kGroupBegin = kCameraBegin + kMaxCameraIntrinsics;
constexpr int kFeatureBegin = kGroupBegin + kGroupSize * kMaxGroup;
constexpr int kFullSize = kFeatureBegin + kFeatureSize * kMaxFeature;
////////////////////////////////////////
// STATE
////////////////////////////////////////
struct State {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // nominal state
  SO3 Rsb;       // body to spatial rotation
  Vec3 Tsb, Vsb; // body to spatial translation and velocity
  Vec3 bg, ba;   // gyro and accl bias

  SO3 Rbc;
  Vec3 Tbc;
  SO3 Rg;

  number_t td;

  using Tangent = Eigen::Matrix<number_t, kMotionSize, 1>;
  State &operator+=(const Tangent &dX) {
    Rsb *= SO3::exp(dX.segment<3>(Index::Wsb));
    Tsb += dX.segment<3>(Index::Tsb);
    Vsb += dX.segment<3>(Index::Vsb);
    bg += dX.segment<3>(Index::bg);
    ba += dX.segment<3>(Index::ba);
    Rbc *= SO3::exp(dX.segment<3>(Index::Wbc));
    Tbc += dX.segment<3>(Index::Tbc);
    // Rg *= SO3::exp(Vec3{dX(Index::Wg), dX(Index::Wg + 1), 0.0});
    Rg *= SO3::exp(dX.segment<3>(Index::Wg));
// std::cout << "Wg=" << dX.segment<3>(Index::Wg).transpose() << std::endl;
#ifdef USE_ONLINE_TEMPORAL_CALIB
    td += dX(Index::td);
#endif
    return *this;
  }
};

State operator-(const State &s1, const State &s2);
std::ostream &operator<<(std::ostream &out, const State &s);
////////////////////////////////////////
// STATUS
////////////////////////////////////////
enum class TrackStatus : int {
  CREATED = 0,  // feature just been detected
  TRACKED = 1,  // feature being tracked well
  REJECTED = 2, // rejected by the tracker
  DROPPED = 3   // out of view?
};
enum class FeatureStatus : int {
  CREATED = 0,
  INITIALIZING = 1,
  READY = 2,
  INSTATE = 3,
  REJECTED_BY_FILTER = 4,
  REJECTED_BY_TRACKER = 5,
  DROPPED = 6,
  // GAUGE = 7 // chosen to fix gauge freedom
};

enum class GroupStatus : int {
  CREATED = 0,  // newly created
  INSTATE = 1,  // instate
  FLOATING = 2, // floating
  GAUGE = 3     // chosen to fix gauge freedom
};

class Feature;
using FeaturePtr = Feature *;
class Group;
using GroupPtr = Group *;
class Tracker;
using TrackerPtr = Tracker *;
using Camera = CameraManager;
using CameraPtr = Camera *;
using Config = Json::Value;
class MemoryManager;
using MemoryManagerPtr = MemoryManager *;
class Estimator;
////////////////////////////////////////
// CUSTOM EXCEPTION
////////////////////////////////////////
struct NotImplemented : public std::exception {
  virtual char const *what() noexcept { return "NOT implemented"; }
};

} // namespace feh
