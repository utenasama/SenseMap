//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#include "match_data_container.h"
#include <fstream>

namespace sensemap{

MatchDataContainer::~MatchDataContainer(){
    results_.clear();
}
void MatchDataContainer::Clear(){
    results_.clear();
}

void MatchDataContainer::AddMatch(const FeatureMatcherData& match){
    results_.emplace_back(match);
}

const std::vector<FeatureMatcherData>& MatchDataContainer::GetMatchResults(){
    return results_;
}

void MatchDataContainer::WriteMatch(const std::string& path){
    std::ofstream file(path, std::ofstream::binary);
    CHECK(file.is_open());

    file<<results_.size()<<std::endl;

    for(auto result:results_){
        file<<result.image_id1<<" "<<result.image_id2<<std::endl;
        
        //write matches
        file<<result.matches.size()<<std::endl;
        for(auto match: result.matches){
            file<<match.point2D_idx1<<" "<<match.point2D_idx2<<std::endl;
        }
        
        //write inliermatches in two view geometry
        file<<result.two_view_geometry.inlier_matches.size();
        for(auto match:result.two_view_geometry.inlier_matches){
            file<<match.point2D_idx1<<" "<<match.point2D_idx2<<std::endl;
        }

        //write config
        file<<result.two_view_geometry.config<<std::endl;

        //write E, F, H
        for(unsigned int i=0;i<3;++i){
            for(unsigned int j=0;j<3;++j){
                file<<result.two_view_geometry.E(i,j)<<" ";
            }
        }
        file<<std::endl;

        for(unsigned int i=0;i<3;++i){
            for(unsigned int j=0;j<3;++j){
                file<<result.two_view_geometry.F(i,j)<<" ";
            }
        }
        file<<std::endl;

        for(unsigned int i=0;i<3;++i){
            for(unsigned int j=0;j<3;++j){
                file<<result.two_view_geometry.H(i,j)<<" ";
            }
        }
        file<<std::endl;

        //write relateive pose
        for(unsigned int i=0;i<4;++i){
            file<<result.two_view_geometry.qvec(i)<<" ";
        }
        file<<std::endl;

        for(unsigned int i=0;i<3;++i){
            file<<result.two_view_geometry.tvec(i)<<" ";
        }
        file<<std::endl;
    }

    file.close();
}

void MatchDataContainer::ReadMatch(const std::string& path){
    std::ifstream file(path, std::ofstream::binary);
    CHECK(file.is_open());

    size_t pair_count;
    file>>pair_count;

    while(pair_count--){
        FeatureMatcherData fmdata;
        file>>fmdata.image_id1>>fmdata.image_id2;

        size_t match_count;
        file>>match_count;
        while(match_count--){
            FeatureMatch match;
            file>>match.point2D_idx1>>match.point2D_idx2;
            fmdata.matches.emplace_back(match);
        }

        size_t inlier_match_count;
        file>>inlier_match_count;
        while(inlier_match_count--){
            FeatureMatch match;
            file>>match.point2D_idx1>>match.point2D_idx2;
            fmdata.two_view_geometry.inlier_matches.emplace_back(match);
        }

        file>>fmdata.two_view_geometry.config;

        //read E, F, H
        for(unsigned int i=0;i<3;++i){
            for(unsigned int j=0;j<3;++j){
                file>>fmdata.two_view_geometry.E(i,j);
            }
        }
        for(unsigned int i=0;i<3;++i){
            for(unsigned int j=0;j<3;++j){
                file>>fmdata.two_view_geometry.F(i,j);
            }
        }
        for(unsigned int i=0;i<3;++i){
            for(unsigned int j=0;j<3;++j){
                file>>fmdata.two_view_geometry.H(i,j);
            }
        }
        //read relative pose
        for(unsigned int i=0;i<4;++i){
            file>>fmdata.two_view_geometry.qvec(i);
        }
        for(unsigned int i=0;i<3;++i){
            file>>fmdata.two_view_geometry.tvec(i);
        }
        results_.emplace_back(fmdata);
    }
}
} //namespace sensemap
