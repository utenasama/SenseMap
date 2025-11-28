#ifndef SENSEMAP_UTIL_CERES_TYPES_H_
#define SENSEMAP_UTIL_CERES_TYPES_H_

#include <ceres/ceres.h>

#include <functional>

namespace ceres {
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 2
    typedef ceres::Manifold LocalParameterization;
    typedef ceres::QuaternionManifold QuaternionParameterization;
    typedef ceres::SubsetManifold SubsetParameterization;
    typedef ceres::SphereManifold<ceres::DYNAMIC> SphereParameterization;
#else
    typedef ceres::LocalParameterization LocalParameterization;
    typedef ceres::QuaternionParameterization QuaternionParameterization;
    typedef ceres::SubsetParameterization SubsetParameterization;
    typedef ceres::HomogeneousVectorParameterization SphereParameterization;
#endif
}

#endif