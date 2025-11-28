//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_UTIL_MAT_H_
#define SENSEMAP_UTIL_MAT_H_

#include <fstream>
#include <string>
#include <vector>

#include "util/endian.h"
#include "util/logging.h"

namespace sensemap {

template <typename T>
class Mat {
 public:
  Mat();
  Mat(const size_t width, const size_t height, const size_t depth);

  bool IsValid() const;

  size_t GetWidth() const;
  size_t GetHeight() const;
  size_t GetDepth() const;

  size_t GetNumBytes() const;

  T Get(const size_t row, const size_t col, const size_t slice = 0) const;
  T Get(const uint64_t idx) const;
  float GetInterpolation(
    const float y, const float x, const size_t slice = 0) const;
  void GetSlice(const size_t row, const size_t col, T* values) const;
  T* GetPtr();
  const T* GetPtr() const;

  const std::vector<T>& GetData() const;

  void Set(const size_t row, const size_t col, const T value);
  void Set(const size_t row, const size_t col, const size_t slice,
           const T value);
  void Set(const uint64_t idx, const T value);
  void SetSlice(const size_t row, const size_t col, const T* values);
  void Set(const std::vector<T>& data);

  void Fill(const T value);

  void Read(const std::string& path);
  void Write(const std::string& path) const;

 protected:
  size_t width_ = 0;
  size_t height_ = 0;
  size_t depth_ = 0;
  std::vector<T> data_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
Mat<T>::Mat() : Mat(0, 0, 0) {}

template <typename T>
Mat<T>::Mat(const size_t width, const size_t height, const size_t depth)
    : width_(width), height_(height), depth_(depth) {
  data_.resize(width_ * height_ * depth_, 0);
}

template <typename T>
bool Mat<T>::IsValid() const {
  return width_ != 0 && height_ != 0 && depth_ != 0;
}

template <typename T>
size_t Mat<T>::GetWidth() const {
  return width_;
}

template <typename T>
size_t Mat<T>::GetHeight() const {
  return height_;
}

template <typename T>
size_t Mat<T>::GetDepth() const {
  return depth_;
}

template <typename T>
size_t Mat<T>::GetNumBytes() const {
  return data_.size() * sizeof(T);
}

template <typename T>
T Mat<T>::Get(const size_t row, const size_t col, const size_t slice) const {
  return data_.at(slice * width_ * height_ + row * width_ + col);
}

template <typename T>
T Mat<T>::Get(const uint64_t idx) const {
  return data_.at(idx);
}

template <typename T>
float Mat<T>::GetInterpolation(
  const float y, const float x, const size_t slice) const {
  const float lx((int)x);
  const float ly((int)y);
  const float w0(x - lx), w1(1.0f - w0);
  const float w2(y - ly), w3(1.0f - w2);

  const float val0 = Get(y, x, slice);
  const float val1 = Get(y, x + 1, slice);
  const float val2 = Get(y + 1, x, slice);
  const float val3 = Get(y + 1, x + 1, slice);

  return (val0 * w1 + val1 * w0) * w3 + (val2 * w1 + val3 * w0) * w2;
}

template <typename T>
void Mat<T>::GetSlice(const size_t row, const size_t col, T* values) const {
  for (size_t slice = 0; slice < depth_; ++slice) {
    values[slice] = Get(row, col, slice);
  }
}

template <typename T>
T* Mat<T>::GetPtr() {
  return data_.data();
}

template <typename T>
const T* Mat<T>::GetPtr() const {
  return data_.data();
}

template <typename T>
const std::vector<T>& Mat<T>::GetData() const {
  return data_;
}

template <typename T>
void Mat<T>::Set(const size_t row, const size_t col, const T value) {
  Set(row, col, 0, value);
}

template <typename T>
void Mat<T>::Set(const size_t row, const size_t col, const size_t slice,
                 const T value) {
  data_.at(slice * width_ * height_ + row * width_ + col) = value;
}

template <typename T>
void Mat<T>::Set(const uint64_t idx, const T value) {
  data_.at(idx) = value;
}

template <typename T>
void Mat<T>::SetSlice(const size_t row, const size_t col, const T* values) {
    for (size_t slice = 0; slice < depth_; ++slice) {
      Set(row, col, slice, values[slice]);
    }
}

template <typename T>
void Mat<T>::Set(const std::vector<T>& data) {
  data_ = data;
}

template <typename T>
void Mat<T>::Fill(const T value) {
  std::fill(data_.begin(), data_.end(), value);
}

template <typename T>
void Mat<T>::Read(const std::string& path) {
  std::fstream text_file(path, std::ios::in | std::ios::binary);
  CHECK(text_file.is_open()) << path;

  char unused_char;
  text_file >> width_ >> unused_char >> height_ >> unused_char >> depth_ >>
      unused_char;
  std::streampos pos = text_file.tellg();
  text_file.close();

  // CHECK_GT(width_, 0);
  // CHECK_GT(height_, 0);
  // CHECK_GT(depth_, 0);
  data_.resize(width_ * height_ * depth_);

  std::fstream binary_file(path, std::ios::in | std::ios::binary);
  CHECK(binary_file.is_open()) << path;
  binary_file.seekg(pos);
  ReadBinaryLittleEndian<T>(&binary_file, &data_);
  // binary_file.read((char *)&data_[0], sizeof(T) * width_ * height_ * depth_);
  binary_file.close();
}

template <typename T>
void Mat<T>::Write(const std::string& path) const {
  std::fstream text_file(path, std::ios::out);
  CHECK(text_file.is_open()) << path;
  text_file << width_ << "&" << height_ << "&" << depth_ << "&";
  text_file.close();

  std::fstream binary_file(path,
                           std::ios::out | std::ios::binary | std::ios::app);
  CHECK(binary_file.is_open()) << path;
  // binary_file.write((char *)data_.data(), sizeof(T) * width_ * height_ * depth_);
  WriteBinaryLittleEndian<T>(&binary_file, data_);
  binary_file.close();
}

void DownsampleImage(const float* data, const int rows, const int cols,
                     const int new_rows, const int new_cols,
                     float* downsampled);

void InterpolateImage(const float* data, const int rows, const int cols,
                      const int new_rows, const int new_cols,
                      float* downsampled);

typedef Mat<unsigned char> MatXu;
typedef Mat<unsigned short> MatXs;
typedef Mat<int> MatXi;
typedef Mat<unsigned int> MatXui;
typedef Mat<float> MatXf;
typedef Mat<double> MatXd;

}  // namespace sensemap

#endif  // SENSEMAP_UTIL_MAT_H_
