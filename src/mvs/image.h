//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_MVS_IMAGE_H_
#define SENSEMAP_MVS_IMAGE_H_

#include <cstdint>
#include <fstream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "util/bitmap.h"

namespace sensemap {
namespace mvs {

class Image {
 public:
  Image();
  Image(const std::string& path, const size_t width, const size_t height,
        const float* K, const float* R, const float* T, 
        const bool is_from_rig = false, const bool is_from_lidar = false);

  ~Image();

  inline size_t GetWidth() const;
  inline size_t GetHeight() const;

  void SetBitmap(const Bitmap& bitmap);
  inline Bitmap& GetBitmap();
  inline const Bitmap& GetBitmap() const;

  inline const std::string& GetPath() const;
  inline const float* GetR() const;
  inline const float* GetT() const;
  inline const float* GetC() const;
  inline const float* GetK() const;
  inline const float* GetP() const;
  inline const float* GetInvP() const;
  inline const float* GetViewingDirection() const;

  inline bool IsRig() const;
  inline bool IsLidar() const;


  void Rescale(const float factor);
  void Rescale(const float factor_x, const float factor_y);
  void Downsize(const size_t max_width, const size_t max_height);

 private:
  std::string path_;
  size_t width_;
  size_t height_;
  float K_[9];
  float R_[9];
  float T_[3];
  float C_[3];
  float P_[12];
  float inv_P_[12];
  Bitmap bitmap_;
  bool is_rig;
  bool is_lidar;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

size_t Image::GetWidth() const { return width_; }

size_t Image::GetHeight() const { return height_; }

Bitmap& Image::GetBitmap() { return bitmap_; }

const Bitmap& Image::GetBitmap() const { return bitmap_; }

const std::string& Image::GetPath() const { return path_; }

const float* Image::GetR() const { return R_; }

const float* Image::GetT() const { return T_; }

const float* Image::GetC() const { return C_; }

const float* Image::GetK() const { return K_; }

const float* Image::GetP() const { return P_; }

const float* Image::GetInvP() const { return inv_P_; }

const float* Image::GetViewingDirection() const { return &R_[6]; }

bool Image::IsRig() const { return is_rig; }
bool Image::IsLidar() const { return is_lidar; }

}  // namespace mvs
}  // namespace sensemap

#endif  // SENSEMAP_MVS_IMAGE_H_
