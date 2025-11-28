//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#ifndef SENSEMAP_ESTIMATORS_MOTION_AVERAGE_TRANSLATION_AVERAGE_H_
#define SENSEMAP_ESTIMATORS_MOTION_AVERAGE_TRANSLATION_AVERAGE_H_

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "util/alignment.h"
namespace sensemap{

struct TranslationAveragerOptions {
    // Maximal iteration of trianglation
    int max_iteration_triangulation=10;
};

//This class handles the plain average of a set of translations, as well as the 
//triangulation from a set of relative translations and anchor translations. 
//The implementation of the triangulation follows this paper:
//       
//   Venu Madhav Govindu, Combining Two-view Constraints for Motion Estimation,
//   CVPR 2005
//    
class TranslationAverager {
public:
    TranslationAverager(TranslationAveragerOptions& options);

    Eigen::Vector3d Triangulate(const std::vector<Eigen::Vector3d>& t_relative,
                                const std::vector<Eigen::Vector3d>& t_anchor);

    Eigen::Vector3d Mean(const std::vector<Eigen::Vector3d>& translations);

private:
    TranslationAveragerOptions options_;
};


//The translaton triangulation estimator used in RANSAC
class TranslationTriangulationEstimator{

public:
	static const int kMinNumSamples = 2;

  	typedef Eigen::Vector3d X_t;
  	typedef Eigen::Vector3d Y_t;
  	typedef Eigen::Vector3d M_t; 

	std::vector<M_t> Estimate(const std::vector<X_t>& t_relative,
							  const std::vector<Y_t>& t_anchor);

	void Residuals(const std::vector<X_t>& t_relative,
				   const std::vector<Y_t>& t_anchor, const M_t& t_absolute,
				   std::vector<double>* residuals); 

};

//The translation average estimator used in RANSC
class TranslationAverageEstimator{
public:
	static const int kMinNumSamples = 1;

  	typedef Eigen::Vector3d X_t;
  	typedef Eigen::Vector3d Y_t;
  	typedef Eigen::Vector3d M_t; 

	std::vector<M_t> Estimate(const std::vector<X_t>& src,
							  const std::vector<Y_t>& dst);

	void Residuals(const std::vector<X_t>& src,
				   const std::vector<Y_t>& dst, const M_t& t_esimated,
				   std::vector<double>* residuals); 

};

}//namespace sensemap

#endif //SENSEMAP_ESTIMATORS_MOTION_AVERAGE_TRANSLATION_AVERAGE_H_