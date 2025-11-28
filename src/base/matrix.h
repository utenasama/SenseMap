//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_UTIL_MATRIX_H_
#define SENSEMAP_UTIL_MATRIX_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>
#include "util/types.h"
#include "util/misc.h"
#include <fstream>

namespace sensemap {

// Check if the given floating point array contains a NaN value.
template <typename Derived>
inline bool IsNaN(const Eigen::MatrixBase<Derived>& x);

// Check if the given floating point array contains infinity.
template <typename Derived>
inline bool IsInf(const Eigen::MatrixBase<Derived>& x);

// Perform RQ decomposition on matrix. The RQ decomposition transforms a matrix
// A into the product of an upper triangular matrix R (also known as
// right-triangular) and an orthogonal matrix Q.
template <typename MatrixType>
void DecomposeMatrixRQ(const MatrixType& A, MatrixType* R, MatrixType* Q);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename Derived>
bool IsNaN(const Eigen::MatrixBase<Derived>& x) {
  return !(x.array() == x.array()).all();
}

template <typename Derived>
bool IsInf(const Eigen::MatrixBase<Derived>& x) {
  return !((x - x).array() == (x - x).array()).all();
}

template <typename MatrixType>
void DecomposeMatrixRQ(const MatrixType& A, MatrixType* R, MatrixType* Q) {
  const MatrixType A_flipud_transpose =
      A.transpose().rowwise().reverse().eval();

  const Eigen::HouseholderQR<MatrixType> QR(A_flipud_transpose);
  const MatrixType& Q0 = QR.householderQ();
  const MatrixType& R0 = QR.matrixQR();

  *R = R0.transpose().colwise().reverse().eval();
  *R = R->rowwise().reverse().eval();
  for (int i = 0; i < R->rows(); ++i) {
    for (int j = 0; j < R->cols() && (R->cols() - j) > (R->rows() - i); ++j) {
      (*R)(i, j) = 0;
    }
  }

  *Q = Q0.transpose().colwise().reverse().eval();

  // Make the decomposition unique by requiring that det(Q) > 0.
  if (Q->determinant() < 0) {
    Q->row(1) *= -1.0;
    R->col(1) *= -1.0;
  }
}

template <typename T>
double InterpolateBilinear(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &matrix, double x, double y) {
    int width = matrix.cols();
    int height = matrix.rows();
    // Image border clipping
	x = std::max(x, 0.0);
	x = std::min(x, (double)(width - 1));
	y = std::max(y, 0.0);
	y = std::min(y, (double)(height - 1));

	int x0 = static_cast<int>(std::floor(x));
	if (x0 == width - 1) {
	    x0--;
	}
	const int x1 = x0 + 1;
	int y0 = static_cast<int>(std::floor(y));
	if (y0 == height - 1) {
	    y0--;
	}
	const int y1 = y0 + 1;

	const double dx = x - x0;
	const double dy = y - y0;
	const double dx_1 = 1 - dx;
	const double dy_1 = 1 - dy;

    // Top row, column-wise linear interpolation.
    const double v0 = dx_1 * matrix(y0, x0) + dx * matrix(y0, x1);

    // Bottom row, column-wise linear interpolation.
    const double v1 = dx_1 * matrix(y1, x0) + dx * matrix(y1, x1);

    // Row-wise linear interpolation.
	return dy_1 * v0 + dy * v1;
}

}  // namespace sensemap

#endif  // SENSEMAP_UTIL_MATRIX_H_
