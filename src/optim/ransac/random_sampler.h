//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_OPTIM_RANSAC_RANDOM_SAMPLER_H_
#define SENSEMAP_OPTIM_RANSAC_RANDOM_SAMPLER_H_

#include "sampler.h"

namespace sensemap {

// Random sampler for RANSAC-based methods.
//
// Note that a separate sampler should be instantiated per thread.
class RandomSampler : public Sampler {
 public:
  explicit RandomSampler(const size_t num_samples);

  void Initialize(const size_t total_num_samples) override;

  size_t MaxNumSamples() override;

  std::vector<size_t> Sample() override;

 private:
  const size_t num_samples_;
  std::vector<size_t> sample_idxs_;
};

}  // namespace sensemap

#endif  // SENSEMAP_OPTIM_RANSAC_RANDOM_SAMPLER_H_
