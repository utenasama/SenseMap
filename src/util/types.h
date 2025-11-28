//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_UTIL_TYPES_H_
#define SENSEMAP_UTIL_TYPES_H_

#include "util/alignment.h"

#ifdef _MSC_VER
#if _MSC_VER >= 1600
#include <cstdint>
#else
typedef __int8 int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned __int8 uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#endif
#elif __GNUC__ >= 3
#include <cstdint>
#endif
#include <memory>


// Define non-copyable or non-movable classes.
#define NON_COPYABLE(class_name)          \
  class_name(class_name const&) = delete; \
  void operator=(class_name const& obj) = delete;
#define NON_MOVABLE(class_name) class_name(class_name&&) = delete;

#include <limits>
#include <Eigen/Core>

namespace Eigen {

typedef Eigen::Matrix<float, 3, 4> Matrix3x4f;
typedef Eigen::Matrix<double, 3, 4> Matrix3x4d;
typedef Eigen::Matrix<uint8_t, 3, 1> Vector3ub;
typedef Eigen::Matrix<uint8_t, 4, 1> Vector4ub;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 7, 1> Vector7d;
typedef Eigen::Matrix<size_t, 3, 1> Vector3sz;

typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> RowMatrix3f;
typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> RowMatrix3d;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> RowMatrix4f;
typedef Eigen::Matrix<double, 4, 4, Eigen::RowMajor> RowMatrix4d;
typedef Eigen::Matrix<float, 3, 4, Eigen::RowMajor> RowMatrix3x4f;
typedef Eigen::Matrix<double, 3, 4, Eigen::RowMajor> RowMatrix3x4d;

typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXi;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXf;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXd;
typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXuc;

}  // namespace Eigen

namespace sensemap {

////////////////////////////////////////////////////////////////////////////////
// Index types, determines the maximum number of objects.
////////////////////////////////////////////////////////////////////////////////

// Unique identifier for cameras.
typedef uint32_t camera_t;

// Unique identifier for local cameras
typedef uint32_t local_camera_t;

// Unique identifier for images.
typedef uint32_t image_t;

// Unique identifier for lidarsweep.
typedef uint32_t sweep_t;

// Unique identifier for image labels.
typedef uint32_t label_t;

// Unique identifier for marker ids.
typedef uint32_t marker_t;

// Each image pair gets a unique ID.
typedef uint64_t image_pair_t;
// Each point pair gets a unique ID
typedef uint64_t point_pair_t;

// Index per image, i.e. determines maximum number of 2D points per image.
typedef uint32_t point2D_t;

// Unique identifier per track.
typedef uint64_t track_t;

// Unique identifier per map point.
typedef uint64_t mappoint_t;

// Unique identifier for scene clusters.
typedef uint32_t cluster_t;

//Each cluster pair gets a unique ID
typedef uint64_t cluster_pair_t;

//Unique identifier for a general view (an image or a scene cluster)
typedef uint32_t view_t;
//identifier for a general view pair
typedef uint64_t view_pair_t;

typedef std::pair<view_t, view_t> ViewIdPair;

// Unique identifier for blocks.
typedef uint32_t block_t;

// Values for invalid identifiers or indices.
const camera_t kInvalidCameraId = std::numeric_limits<camera_t>::max();
const sweep_t kInvalidLidarSweepId = std::numeric_limits<image_t>::max();
const image_t kInvalidImageId = std::numeric_limits<image_t>::max();
const label_t kInvalidLabelId = std::numeric_limits<image_t>::max();
const image_pair_t kInvalidImagePairId =
		std::numeric_limits<image_pair_t>::max();
const point2D_t kInvalidPoint2DIdx = std::numeric_limits<point2D_t>::max();
const mappoint_t kInvalidMapPointId = std::numeric_limits<mappoint_t>::max();

#define EIGEN_STL_UMAP(KEY, VALUE)                                   \
  std::unordered_map<KEY, VALUE, std::hash<KEY>, std::equal_to<KEY>, \
                     Eigen::aligned_allocator<std::pair<KEY const, VALUE> > >

#define EIGEN_STL_MAP(KEY, VALUE)                                   \
  std::map<KEY, VALUE, std::less<KEY>, \
                     Eigen::aligned_allocator<std::pair<KEY const, VALUE> > >



#define SENSEMAP_POINTER_TYPEDEFS(TypeName)             \
  typedef std::shared_ptr<TypeName> Ptr;              \
  typedef std::shared_ptr<const TypeName> ConstPtr;   \
  typedef std::weak_ptr<TypeName> WeakPtr;            \
  typedef std::weak_ptr<const TypeName> WeakConstPtr; \
  void definePointerTypedefs##__FILE__##__LINE__(void)
}  // namespace sensemap

#endif