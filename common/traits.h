// Type traits and TMP facilities.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#include <type_traits>

namespace xivo {
// type traits utilities
template <class T>
using plain = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

} // namespace xivo

