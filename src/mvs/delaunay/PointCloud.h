//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_MVS_POINTCLOUD_H_
#define SENSEMAP_MVS_POINTCLOUD_H_

#include <Eigen/Dense>

#include "AABB.h"
#include "List.h"

namespace sensemap {
namespace mvs {

class PointCloud {
public:
    typedef IDX Index;

    typedef Eigen::Vector3f Point;
    typedef mvs::cList<Point,const Point&,2,8192> PointArr;

    typedef float Score;
    typedef mvs::cList<Score,const Score&,2,8192> ScoreArr;

    typedef uint32_t View;
    typedef mvs::cList<View,const View,0,4,uint32_t> ViewArr;
    typedef mvs::cList<ViewArr> PointViewArr;

    typedef float Weight;
    typedef mvs::cList<Weight,const Weight,0,4,uint32_t> WeightArr;
    typedef mvs::cList<WeightArr> PointWeightArr;

    typedef Eigen::Vector3f Normal;
    typedef CLISTDEF0(Normal) NormalArr;

    typedef Eigen::Matrix<uint8_t, 3, 1> Color;
    typedef CLISTDEF0(Color) ColorArr;

    typedef char PointType;
    typedef mvs::cList<PointType, const PointType&, 2, 8192> PointTypeArr;

    typedef AABB3f Box;

public:
    PointArr points;
    ScoreArr scores;
    PointViewArr pointViews; // array of views for each point (ordered increasing)
    PointWeightArr pointWeights;
    NormalArr normals;
    ColorArr colors;
    PointTypeArr pointTypes;

public:
    inline PointCloud() {}

    void Release();

    inline bool IsEmpty() const { ASSERT(points.GetSize() == pointViews.GetSize() || pointViews.IsEmpty()); return points.IsEmpty(); }
    inline bool IsValid() const { ASSERT(points.GetSize() == pointViews.GetSize() || pointViews.IsEmpty()); return !pointViews.IsEmpty(); }
    inline size_t GetSize() const { ASSERT(points.GetSize() == pointViews.GetSize() || pointViews.IsEmpty()); return points.GetSize(); }

    void RemovePoint(IDX idx);

    Box GetAABB() const;
    Box GetAABB(const Box& bound) const;
    Box GetAABB(unsigned minViews) const;

    // bool Load(const String& fileName);
    // bool Save(const String& fileName, bool bLegacyTypes=false) const;

    // #ifdef _USE_BOOST
    // // implement BOOST serialization
    // template <class Archive>
    // void serialize(Archive& ar, const unsigned int /*version*/) {
    //     ar & points;
    //     ar & pointViews;
    //     ar & pointWeights;
    //     ar & normals;
    //     ar & colors;
    // }
    // #endif
};

} // namespace mvs
} // namespace sensemap

#endif