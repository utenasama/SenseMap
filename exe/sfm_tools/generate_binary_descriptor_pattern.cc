// Copyright (c) 2021, SenseTime Group.
// All rights reserved.

#include "feature/utils.h"

using namespace sensemap;

int main(int argc, char* argv[]){
    std::vector<std::pair<int,int>> binary_patterns;
    
    GenerateDimPiarsForBinarization(binary_patterns,atoi(argv[1]));
    WriteDimPiarsForBinarization(binary_patterns,argv[2]);
    return 0;
}

