// Copyright (c) 2021, SenseTime Group.
// All rights reserved.

#include "feature/utils.h"
#include <iostream>
using namespace sensemap;

int main(int argc, char* argv[]){
    std::vector<std::pair<int,int>> binary_patterns;
    
    ReadDimPiarsForBinarization(binary_patterns,argv[1]);
    
    std::cout<<"pairs count: "<<binary_patterns.size()<<std::endl;
    
    for(size_t i = 0; i<binary_patterns.size(); ++i){
        std::cout<<binary_patterns[i].first<<" "<<binary_patterns[i].second<<std::endl;
    }

    return 0;
}
