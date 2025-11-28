//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "util/random.h"

namespace sensemap {

thread_local std::mt19937* PRNG = nullptr;

void SetPRNGSeed(unsigned seed) {
  // Overwrite existing PRNG
  if (PRNG != nullptr) {
    delete PRNG;
  }

  if (seed == kRandomPRNGSeed) {
    seed = static_cast<unsigned>(
        std::chrono::system_clock::now().time_since_epoch().count());
  }

  PRNG = new std::mt19937(seed);
  srand(seed);
}

}  // namespace sensemap
