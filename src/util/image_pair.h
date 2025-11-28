//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#ifndef SENSEMAP_UTIL_IMAGE_PAIR_H_
#define SENSEMAP_UTIL_IMAGE_PAIR_H_

#include "util/types.h"
#include "util/logging.h"

namespace sensemap{

const  size_t kMaxNumImages =
    static_cast<size_t>(std::numeric_limits<int32_t>::max());

namespace utility {

bool SwapImagePair(const image_t image_id1, const image_t image_id2);

image_pair_t ImagePairToPairId(const image_t image_id1,
                               const image_t image_id2);

void PairIdToImagePair(const image_pair_t pair_id, image_t* image_id1,
                                 image_t* image_id2);

void PairIdToImagePairOrdered(const image_pair_t pair_id, image_t* image_id1,
                                 image_t* image_id2);


}

}//namespace sensemap

#endif //SENSEMAP_UTIL_IMAGE_PAIR_H_