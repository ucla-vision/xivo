#include "core.h"

namespace feh {

// utilities for nominal state
State operator-(const State &s1, const State &s2) {
  return {s1.Rsb * s2.Rsb.inv(), s1.Tsb - s2.Tsb,    s1.Vsb - s2.Vsb,
          s1.bg - s2.bg,         s1.ba - s2.ba,      s1.Rbc * s2.Rbc.inv(),
          s1.Tbc - s2.Tbc,       s1.Rg * s2.Rg.inv()};
}

std::ostream &operator<<(std::ostream &os, const State &s) {
  os << "\n=====\n";
  os << "Rsb=\n" << s.Rsb.matrix();
  os << "\nTsb=\n" << s.Tsb.transpose();
  os << "\nVsb=\n" << s.Vsb.transpose();
  os << "\nbg=\n" << s.bg.transpose();
  os << "\nba=\n" << s.ba.transpose();
  os << "\nRbc=\n" << s.Rbc.matrix();
  os << "\nTbc=\n" << s.Tbc.transpose();
  os << "\nRg=\n" << s.Rg.matrix();
  os << "\n=====\n";
  return os;
}

} // namespace feh
