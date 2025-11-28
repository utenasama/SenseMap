//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#ifndef SENSEMAP_CONTAINER_MATCH_DATA_CONTAINER_H_
#define SENSEMAP_CONTAINER_MATCH_DATA_CONTAINER_H_
#include "util/types.h"
#include "feature/types.h"
#include "estimators/two_view_geometry.h"

namespace sensemap{

struct FeatureMatcherData {
    image_t image_id1 = kInvalidImageId;
    image_t image_id2 = kInvalidImageId;
    FeatureMatches matches;
    TwoViewGeometry two_view_geometry;
};

class MatchDataContainer{
public:
    MatchDataContainer(){}
    ~MatchDataContainer();

    void AddMatch(const FeatureMatcherData& match);
    const std::vector<FeatureMatcherData>& GetMatchResults();
    
    void WriteMatch(const std::string& path);
    void ReadMatch(const std::string& path);
    void Clear();
private:
    std::vector<FeatureMatcherData> results_;
};

}//namespace sensemap


#endif//SENSEMAP_CONTAINER_MATCH_DATA_CONTAINER_H_