
// Copyright (c) 2020, SenseTime Group.
// All rights reserved.


#include "base/reconstruction_manager.h"
#include "estimators/camera_alignment.h"
#include "base/similarity_transform.h"

using namespace sensemap;
int main(int argc, char* argv[]) {


    CameraAlignmentEstimator camera_alignment_estimator;
    std::vector<Eigen::Vector3d> points1;
    std::vector<Eigen::Vector3d> points2;


    // points1.push_back(Eigen::Vector3d(-2182969.566002094,4381167.689267585,4075378.60699821));
    // points1.push_back(Eigen::Vector3d(-2182951.9952697223,4381231.492773284,4075319.823048084));
    // points1.push_back(Eigen::Vector3d(-2183063.468986798, 4381227.111941094, 4075265.1879562833));

    points1.push_back(Eigen::Vector3d(-2183392.2003359366,4380857.49233792,4075484.9438952));
    points1.push_back(Eigen::Vector3d(-2183374.7324027075,4380921.131805537,4075426.2885615323));
    points1.push_back(Eigen::Vector3d(-2183485.9677452655,4380917.141438899,4075371.3530099895));

    points2.push_back(Eigen::Vector3d(197.3822,0, 329.4864));
    points2.push_back(Eigen::Vector3d(154.2612,0, 253.5741));
    points2.push_back(Eigen::Vector3d(254.0309,0,182.6870));


    for(int i = 0; i< 3; i++){
        for(int j = i+1; j<3; j++){
            std::cout<<"distance between point "<<i<<" and point "<<j<<" in ECEF: "<<(points1[i]-points1[j]).norm()<<std::endl;
            std::cout<<"distance between point "<<i<<" and point "<<j<<" in CAD: "<<(points2[i]-points2[j]).norm()<<std::endl;
        }
    }

    std::vector<Eigen::Matrix<double, 3, 4>> refined_transforms = camera_alignment_estimator.Estimate(points1, points2);
    SimilarityTransform3 sim3(refined_transforms[0]);
    std::cout<<"transform scale: "<<sim3.Scale()<<std::endl;

    std::vector<double> residuals;
    residuals.resize(points1.size());
    camera_alignment_estimator.Residuals(points1, points2, refined_transforms[0], &residuals);

    double average_residual = 0.0;
    for (const auto& residual : residuals) {
        average_residual += residual;
    }
    average_residual /= residuals.size();
    std::cout << "average residuals: " << sqrt(average_residual) << " m"<<std::endl;
}