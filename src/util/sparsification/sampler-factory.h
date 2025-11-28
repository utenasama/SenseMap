#ifndef MAP_SPARSIFICATION_SAMPLER_FACTORY_H_
#define MAP_SPARSIFICATION_SAMPLER_FACTORY_H_

#include "sampler-base.h"

namespace sensemap {

SamplerBase::Ptr createSampler(SamplerBase::Type sampler_type);

}  // namespace map_sparsification
#endif  // MAP_SPARSIFICATION_SAMPLER_FACTORY_H_
