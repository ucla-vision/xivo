#include <iostream>
#include "helpers.h"

using namespace feh;

int main() {
  int N = 4;  // state size
  int M = 8;  // measurement size
  VecX r;
  MatX Hf, Hx;
  r = MatX::Random(M, 1);
  Hx = MatX::Random(M, N);

  std::cout << "r=\n" << r.transpose() << std::endl;
  std::cout << "Hx=\n" << Hx << std::endl;
  int rows = QR(r, Hx);
  std::cout << "===== After givens =====\n";
  std::cout << "r=\n";
  std::cout << r.head(rows).transpose() << std::endl;
  std::cout << "TH=\n";
  std::cout << Hx.topRows(rows) << std::endl;
}
