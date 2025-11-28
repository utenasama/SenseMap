//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_UTIL_PLY_H_
#define SENSEMAP_UTIL_PLY_H_

#include <string>
#include <vector>

#include "types.h"

namespace sensemap {

struct PlyPoint {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float nx = 0.0f;
  float ny = 0.0f;
  float nz = 0.0f;
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
  uint8_t s_id = -1;
};

struct PlyMeshVertex {
  PlyMeshVertex() : x(0), y(0), z(0), nx(0), ny(0), nz(0), r(0), g(0), b(0) {}
  PlyMeshVertex(const float x, const float y, const float z)
      : x(x), y(y), z(z) {}
  PlyMeshVertex(const float x, const float y, const float z,
                const float nx, const float ny, const float nz)
        : x(x), y(y), z(z), nx(nx), ny(ny), nz(nz) {}
  PlyMeshVertex(const float x, const float y, const float z,
                const float nx, const float ny, const float nz,
                const uint8_t r, const uint8_t g, const uint8_t b)
        : x(x), y(y), z(z), nx(nx), ny(ny), nz(nz), r(r), g(g), b(b) {}

  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float nx = 0.0f;
  float ny = 0.0f;
  float nz = 0.0f;
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
};

struct PlyMeshFace {
  PlyMeshFace() : vertex_idx1(0), vertex_idx2(0), vertex_idx3(0) {}
  PlyMeshFace(const size_t vertex_idx1, const size_t vertex_idx2,
              const size_t vertex_idx3)
      : vertex_idx1(vertex_idx1),
        vertex_idx2(vertex_idx2),
        vertex_idx3(vertex_idx3) {}

  size_t vertex_idx1 = 0;
  size_t vertex_idx2 = 0;
  size_t vertex_idx3 = 0;
};

struct PlyMesh {
  std::vector<PlyMeshVertex> vertices;
  std::vector<PlyMeshFace> faces;
};

bool ReadPointRealType(const std::string& path);

// Read PLY point cloud from text or binary file.
std::vector<PlyPoint> ReadPly(const std::string& path);

// Write PLY point cloud to text or binary file.
void WriteTextPlyPoints(const std::string& path,
                        const std::vector<PlyPoint>& points,
                        const bool write_normal = true,
                        const bool write_rgb = true);

void WriteBinaryPlyPoints(const std::string& path,
                          const std::vector<PlyPoint>& points,
                          const bool write_normal = true,
                          const bool write_rgb = true);

void AppendWriteBinaryPlyPoints(const std::string& path,
                                const std::vector<PlyPoint>& points);

void ReadPly(const std::string& path,
            std::vector<double>& Xs,
            std::vector<double>& Ys,
            std::vector<double>& Zs,
            std::vector<double>& nXs,
            std::vector<double>& nYs,
            std::vector<double>& nZs,
            std::vector<uint8_t>& rs,
            std::vector<uint8_t>& gs,
            std::vector<uint8_t>& bs);

void WriteBinaryPlyPoints(const std::string& path,
                        const std::vector<double>& Xs,
                        const std::vector<double>& Ys,
                        const std::vector<double>& Zs,
                        const std::vector<double>& nXs,
                        const std::vector<double>& nYs,
                        const std::vector<double>& nZs,
                        const std::vector<uint8_t>& rs,
                        const std::vector<uint8_t>& gs,
                        const std::vector<uint8_t>& bs,
                        const bool write_normal = true, 
                        const bool write_rgb = true);

void ReadPointsSemantic(const std::string& path,
                        std::vector<PlyPoint>& points,
                        const bool read_binary = false);

void WritePointsSemantic(const std::string& path,
                         const std::vector<PlyPoint>& points,
                         const bool write_binary = false);

void AppendWritePointsSemantic(const std::string& path,
                               const std::vector<PlyPoint>& points,
                               const bool write_binary = false);
                         
void WritePointsSemanticColor(const std::string& path,
                              const std::vector<PlyPoint>& points);
                       
void AppendWritePointsSemanticColor(const std::string& path,
                              const std::vector<PlyPoint>& points);

void ReadPointsVisibility(
    const std::string& path,
    std::vector<std::vector<uint32_t> > &points_visibility);

// Write the visiblity information into a binary file of the following format:
//
//    <num_points : uint64_t>
//    <num_visible_images_for_point1 : uint32_t>
//    <point1_image_idx1 : uint32_t><point1_image_idx2 : uint32_t> ...
//    <num_visible_images_for_point2 : uint32_t>
//    <point2_image_idx2 : uint32_t><point2_image_idx2 : uint32_t> ...
//    ...
//
// Note that an image_idx in the case of the mvs::StereoFuser does not
// correspond to the image_id of a Reconstruction, but the index of the image in
// the mvs::Model, which is the location of the image in the images.bin/.txt.
void WritePointsVisibility(
    const std::string& path,
    const std::vector<std::vector<uint32_t> > &points_visibility);

void AppendWritePointsVisibility(
    const std::string& path,
    const std::vector<std::vector<uint32_t> > &points_visibility);

void ReadPointsWeight(
    const std::string& path,
    std::vector<std::vector<float> > &points_weight);

void WritePointsWeight(
    const std::string& path,
    const std::vector<std::vector<float> > &points_weight);

void AppendWritePointsWeight(
    const std::string& path,
    const std::vector<std::vector<float> > &points_weight);

void ReadPointsScore(const std::string& path,
                     std::vector<float>& points_score);

void WritePointsScore(const std::string& path,
                      const std::vector<float>& points_score);

void AppendWWritePointsScore(const std::string& path,
                      const std::vector<float>& points_score);

// Write PLY mesh to text or binary file.
void ReadBinaryPlyMesh(const std::string& path, PlyMesh& mesh);

void WriteTextPlyMesh(const std::string& path, const PlyMesh& mesh, 
                      const bool write_as_normal = false,
                      const bool write_as_rgb = false);
void WriteBinaryPlyMesh(const std::string& path, const PlyMesh& mesh, 
                      const bool write_as_normal = false,
                      const bool write_as_rgb = false);

}  // namespace sensemap

#endif  // SENSEMAP_UTIL_PLY_H_
