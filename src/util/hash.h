//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_UTIL_HASH_H_
#define SENSEMAP_UTIL_HASH_H_

#include <boost/functional/hash.hpp>
#include <Eigen/Core>

namespace sensemap {

inline std::size_t HashValue(int x, int y) {
    std::size_t seed = 0;
    std::size_t ux = INT_MAX + x;
    std::size_t uy = INT_MAX + y;
    boost::hash_combine(seed, ux);
    boost::hash_combine(seed, uy);
    return seed;
}

inline std::size_t HashValue(int x, int y, int z) {
    std::size_t seed = 0;
    std::size_t ux = INT_MAX + x;
    std::size_t uy = INT_MAX + y;
    std::size_t uz = INT_MAX + z;
    boost::hash_combine(seed, ux);
    boost::hash_combine(seed, uy);
    boost::hash_combine(seed, uz);
    return seed;
}

inline std::size_t HashValue(Eigen::Vector2i p) {
    std::size_t seed = 0;
    std::size_t ux = INT_MAX + p.x();
    std::size_t uy = INT_MAX + p.y();
    boost::hash_combine(seed, ux);
    boost::hash_combine(seed, uy);
    return seed;
}

inline std::size_t HashValue(Eigen::Vector3i p) {
    std::size_t seed = 0;
    std::size_t ux = INT_MAX + p.x();
    std::size_t uy = INT_MAX + p.y();
    std::size_t uz = INT_MAX + p.z();
    boost::hash_combine(seed, ux);
    boost::hash_combine(seed, uy);
    boost::hash_combine(seed, uz);
    return seed;
}
} // namespace

#endif