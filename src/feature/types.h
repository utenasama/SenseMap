//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_FEATURE_TYPES_H_
#define SENSEMAP_FEATURE_TYPES_H_

#include <vector>

#include <Eigen/Core>

#include "util/types.h"

namespace sensemap {

struct FeatureKeypoint {
  FeatureKeypoint();
  FeatureKeypoint(const float x, const float y);
  FeatureKeypoint(const float x, const float y, const float scale,
                  const float orientation);
  FeatureKeypoint(const float x, const float y, const float a11,
                  const float a12, const float a21, const float a22);

  static FeatureKeypoint FromParameters(const float x, const float y,
                                        const float scale_x,
                                        const float scale_y,
                                        const float orientation,
                                        const float shear);

  // Rescale the feature location and shape size by the given scale factor.
  void Rescale(const float scale);
  void Rescale(const float scale_x, const float scale_y);

  // Compute similarity shape parameters from affine shape.
  float ComputeScale() const;
  float ComputeScaleX() const;
  float ComputeScaleY() const;
  float ComputeOrientation() const;
  float ComputeShear() const;

  // Location of the feature, with the origin at the upper left image corner,
  // i.e. the upper left pixel has the coordinate (0.5, 0.5).
  float x;
  float y;

  // Affine shape of the feature.
  float a11;
  float a12;
  float a21;
  float a22;
  
};

typedef Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor>
    FeatureDescriptor;

struct FeatureMatch {
  FeatureMatch()
      : point2D_idx1(kInvalidPoint2DIdx), point2D_idx2(kInvalidPoint2DIdx) {}
  FeatureMatch(const point2D_t point2D_idx1, const point2D_t point2D_idx2)
      : point2D_idx1(point2D_idx1), point2D_idx2(point2D_idx2) {}

  // Feature index in first image.
  point2D_t point2D_idx1 = kInvalidPoint2DIdx;

  // Feature index in second image.
  point2D_t point2D_idx2 = kInvalidPoint2DIdx;
};

struct PanoramaIndex {
  // Localtion of the feature in the sub-image (only valid for panorama or camera rig)
  int sub_image_id;
  float sub_x;
  float sub_y;
};

struct PieceIndex{
  int piece_id;
  float piece_x;
  float piece_y;
};

class AprilTagDetection{
public:
  int id = -1;
  int local_camera_id = -1;
  std::pair<float,float> p[4];
  std::pair<float,float> cxy;
};

typedef std::vector<FeatureKeypoint> FeatureKeypoints;
typedef std::vector<PanoramaIndex> PanoramaIndexs;
typedef std::vector<PieceIndex> PieceIndexs;

typedef std::vector<AprilTagDetection> AprilTagDetections;
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    FeatureDescriptors;

typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    CompressedFeatureDescriptors;


typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    BinaryFeatureDescriptors;

typedef std::vector<FeatureMatch> FeatureMatches;

} // namespace sensemap

#endif //SENSEMAP_FEATURE_TYPES_H
