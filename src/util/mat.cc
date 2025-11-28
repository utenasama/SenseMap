#include <cmath>
#include "VLFeat/imopv.h"

#include "mat.h"

namespace sensemap {

namespace {
float GetPixelConstantBorder(const float* data, const int rows, const int cols,
                             const int row, const int col) {
  if (row >= 0 && col >= 0 && row < rows && col < cols) {
    return data[row * cols + col];
  } else {
    return 0;
  }
}

void ResampleImageBilinear(const float* data, const int rows, const int cols,
                           const int new_rows, const int new_cols,
                           float* resampled) {
  CHECK_NOTNULL(data);
  CHECK_NOTNULL(resampled);
  CHECK_GT(rows, 0);
  CHECK_GT(cols, 0);
  CHECK_GT(new_rows, 0);
  CHECK_GT(new_cols, 0);

  const float scale_r = static_cast<float>(rows) / static_cast<float>(new_rows);
  const float scale_c = static_cast<float>(cols) / static_cast<float>(new_cols);

#pragma omp parallel for
  for (int r = 0; r < new_rows; ++r) {
    const float r_i = (r + 0.5f) * scale_r - 0.5f;
    const int r_i_min = std::floor(r_i);
    const int r_i_max = r_i_min + 1;
    const float d_r_min = r_i - r_i_min;
    const float d_r_max = r_i_max - r_i;

    for (int c = 0; c < new_cols; ++c) {
      const float c_i = (c + 0.5f) * scale_c - 0.5f;
      const int c_i_min = std::floor(c_i);
      const int c_i_max = c_i_min + 1;
      const float d_c_min = c_i - c_i_min;
      const float d_c_max = c_i_max - c_i;

      // Interpolation in column direction.
      const float value1 =
          d_c_max * GetPixelConstantBorder(data, rows, cols, r_i_min, c_i_min) +
          d_c_min * GetPixelConstantBorder(data, rows, cols, r_i_min, c_i_max);
      const float value2 =
          d_c_max * GetPixelConstantBorder(data, rows, cols, r_i_max, c_i_min) +
          d_c_min * GetPixelConstantBorder(data, rows, cols, r_i_max, c_i_max);

      // Interpolation in row direction.
      resampled[r * new_cols + c] = d_r_max * value1 + d_r_min * value2;
    }
  }
}

void ResampleImageNearest(const float* data, const int rows, const int cols,
                           const int new_rows, const int new_cols,
                           float* resampled) {
  CHECK_NOTNULL(data);
  CHECK_NOTNULL(resampled);
  CHECK_GT(rows, 0);
  CHECK_GT(cols, 0);
  CHECK_GT(new_rows, 0);
  CHECK_GT(new_cols, 0);

  const float scale_r = static_cast<float>(rows) / static_cast<float>(new_rows);
  const float scale_c = static_cast<float>(cols) / static_cast<float>(new_cols);

#pragma omp parallel for
  for (int r = 0; r < new_rows; ++r) {
    const float r_i = (r + 0.5f) * scale_r - 0.5f;

    for (int c = 0; c < new_cols; ++c) {
      const float c_i = (c + 0.5f) * scale_c - 0.5f;

      // Interpolation in column direction.
      const float value = GetPixelConstantBorder(data, rows, cols, r_i, c_i);
      // Interpolation in row direction.
      resampled[r * new_cols + c] = value;
    }
  }
}


void SmoothImage(const float* data, const int rows, const int cols,
                 const float sigma_r, const float sigma_c, float* smoothed) {
  CHECK_NOTNULL(data);
  CHECK_NOTNULL(smoothed);
  CHECK_GT(rows, 0);
  CHECK_GT(cols, 0);
  CHECK_GT(sigma_r, 0);
  CHECK_GT(sigma_c, 0);
  vl_imsmooth_f(smoothed, cols, data, cols, rows, cols, sigma_c, sigma_r);
}
}

void DownsampleImage(const float* data, const int rows, const int cols,
                     const int new_rows, const int new_cols,
                     float* downsampled) {
  CHECK_NOTNULL(data);
  CHECK_NOTNULL(downsampled);
  CHECK_LE(new_rows, rows);
  CHECK_LE(new_cols, cols);
  CHECK_GT(rows, 0);
  CHECK_GT(cols, 0);
  CHECK_GT(new_rows, 0);
  CHECK_GT(new_cols, 0);

  const float scale_c = static_cast<float>(cols) / static_cast<float>(new_cols);
  const float scale_r = static_cast<float>(rows) / static_cast<float>(new_rows);

  const float kSigmaScale = 0.5f;
  const float sigma_c = std::max(std::numeric_limits<float>::epsilon(),
                                 kSigmaScale * (scale_c - 1));
  const float sigma_r = std::max(std::numeric_limits<float>::epsilon(),
                                 kSigmaScale * (scale_r - 1));

  std::vector<float> smoothed(rows * cols);
  SmoothImage(data, rows, cols, sigma_r, sigma_c, smoothed.data());

  ResampleImageBilinear(smoothed.data(), rows, cols, new_rows, new_cols,
                        downsampled);
}

void InterpolateImage(const float* data, const int rows, const int cols,
                      const int new_rows, const int new_cols,
                      float* downsampled) {
  const float scale_c = static_cast<float>(cols) / static_cast<float>(new_cols);
  const float scale_r = static_cast<float>(rows) / static_cast<float>(new_rows);

  ResampleImageBilinear(data, rows, cols, new_rows, new_cols, downsampled);
  // ResampleImageNearest(data, rows, cols, new_rows, new_cols, downsampled);
}

} // namespace sensemap
