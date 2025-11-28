//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#include "util/image_pair.h"

namespace sensemap{

namespace utility {

bool SwapImagePair(const image_t image_id1, const image_t image_id2) {
    return image_id1 > image_id2;
}

image_pair_t ImagePairToPairId(const image_t image_id1,
                               const image_t image_id2) {
    CHECK_GE(image_id1, 0);
    CHECK_GE(image_id2, 0);
    CHECK_LT(image_id1, kMaxNumImages);
    CHECK_LT(image_id2, kMaxNumImages);
    if (SwapImagePair(image_id1, image_id2)) {
        return static_cast<image_pair_t>(kMaxNumImages) * image_id2 + image_id1;
    } 
    else {
        return static_cast<image_pair_t>(kMaxNumImages) * image_id1 + image_id2;
    }
}

void PairIdToImagePair(const image_pair_t pair_id, image_t* image_id1,
                                 image_t* image_id2) {
  *image_id2 = static_cast<image_t>(pair_id % kMaxNumImages);
  *image_id1 = static_cast<image_t>((pair_id - *image_id2) / kMaxNumImages);
  CHECK_GE(*image_id1, 0);
  CHECK_GE(*image_id2, 0);
  CHECK_LT(*image_id1, kMaxNumImages);
  CHECK_LT(*image_id2, kMaxNumImages);
}

void PairIdToImagePairOrdered(const image_pair_t pair_id, image_t* image_id1,
                                 image_t* image_id2){

  image_t id1 = static_cast<image_t>(pair_id % kMaxNumImages);
  image_t id2 = static_cast<image_t>((pair_id - id1) / kMaxNumImages);
  
  *image_id1 = id1<id2?id1:id2;
  *image_id2 = id1<id2?id2:id1;

  CHECK_GE(*image_id1, 0);
  CHECK_GE(*image_id2, 0);
  CHECK_LT(*image_id1, kMaxNumImages);
  CHECK_LT(*image_id2, kMaxNumImages);
}



} // namespace utility

}//namespace sensemap


