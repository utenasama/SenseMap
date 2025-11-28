//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_MVS_NORMAL_MAP_H_
#define SENSEMAP_MVS_NORMAL_MAP_H_

#include <string>
#include <vector>

#include "util/bitmap.h"
#include "util/mat.h"

namespace sensemap {
namespace mvs {

// Normal map class that stores per-pixel normals as a MxNx3 image.
class NormalMap : public Mat<float> {
 public:
  NormalMap();
  NormalMap(const size_t width, const size_t height);
  explicit NormalMap(const Mat<float>& mat);

  void Rescale(const float factor);
  void Rescale(const float factor_x, const float factor_y);
  void Downsize(const size_t max_width, const size_t max_height);

  void Read(const std::string& path);
  void Write(const std::string& path) const;

  void Decode(const MatXui &en_data);
  MatXui Encode() const;
  
  Bitmap ToBitmap() const;
};

}  // namespace mvs
}  // namespace sensemap

#endif  // SENSEMAP_MVS_NORMAL_MAP_H_
