// For readability and maintainability, it's better to adopt
// a modular design approach, where componets like groups and features,
// and even IMUs and cameras whose intrinsic parameters can be estimated,
// maintain their own local nominal state.
// The estimator maintains the overall error state,
// and performs state propagation and measurement update.
// When needed, the estimator pulls local nominal states from each module
// to compute Jacobians, and updates the local nominal states
// by injecting the error state back to the nominal state internal to each
// module.
// We abstract as the notion of ''Componet'' all those modules whose
// internal parameters can be estimated.
//
// We adopt the CRTP idiom to ensure a minimal set of local state
// operations are implemented.
// This is more like to impose design requirements rather than to
// enable (static) polymorphism.
//
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once

namespace feh {

// helper function to check whehter the given "State" type has a "Tangent" type
// defined.
// referene:
//    https://en.cppreference.com/w/cpp/types/void_t
// modified according the has_type_member function
template <class, class = std::void_t<>> struct has_tangent : std::false_type {};

template <class T>
struct has_tangent<T, std::void_t<typename T::Tangent>> : std::true_type {};

// CRTP pattern for static polymorphism
// Derived: Derived class to enforce the implementation of UpdateState
// S: the nominal state for the local parameters.
// Error: the error state for the local parameters.
template <typename Derived, typename State> struct Component {
  std::enable_if_t<has_tangent<State>::value, void>
  UpdateState(const typename State::Tangent &dX) {
    static_cast<Derived *>(this)->UpdateState(dX);
  }

  std::enable_if_t<!has_tangent<State>::value, void>
  UpdateState(const State &dX) {
    static_cast<Derived *>(this)->UpdateState(dX);
  }
};

// example usage: see src/imu.h

} // namespace feh
