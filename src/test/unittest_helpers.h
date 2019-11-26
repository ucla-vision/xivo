#include "alias.h"

using namespace xivo;

void CheckVectorEquality(VecX v1, VecX v2, number_t tol);
void CheckMatrixEquality(MatX M1, MatX M2, number_t tol);
void CheckVecZero(VecX v, number_t tol);
void CheckMatrixZero(MatX M, number_t tol);

Mat3 RandomTransformationMatrix();