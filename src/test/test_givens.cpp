#include <iostream>
#include "helpers.h"

using namespace xivo;

int main() {
  int M = 4;
  VecX r;
  MatX Hf, Hx;
  r = MatX::Random(2 * M, 1);
  Hf = MatX::Random(2 * M, 3);

  Hx = MatX::Random(2 * M, 5);

  std::cout << r.transpose() << std::endl;
  Givens(r, Hx, Hf);
  std::cout << "===== After givens =====\n";
  std::cout << "r=\n";
  std::cout << r.transpose() << std::endl;
  std::cout << "Hf=\n";
  std::cout << Hf << std::endl;

  std::cout << "Hx=\n";
  std::cout << Hx << std::endl;

}
