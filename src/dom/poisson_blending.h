//Copyright (c) 2022, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_DOM_POISSON_BLENDING_H_
#define SENSEMAP_DOM_POISSON_BLENDING_H_

#include <map>

#include <Eigen/Eigen>
#include <Eigen/UmfPackSupport>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>

namespace sensemap {

class PoissonBlender {
public:
    PoissonBlender();
    PoissonBlender(const cv::Mat & mosaic, const cv::Mat & mask, const cv::Mat & drvxy);

    void Solve();

    cv::Mat GetResult();

private:
    bool BuildMatrix(Eigen::SparseMatrix<double> &A, Eigen::Matrix<double, Eigen::Dynamic, 1> &b,
                     Eigen::Matrix<double, Eigen::Dynamic, 1> &u);

    bool PoissonSolve(const Eigen::SparseMatrix<double> &A, const Eigen::Matrix<double, Eigen::Dynamic, 1> &b, 
                      Eigen::Matrix<double, Eigen::Dynamic, 1> &u);

    bool CopyResult(Eigen::Matrix<double, Eigen::Dynamic, 1> &u);
            
private:
    std::map<int, int> mp_;

    cv::Mat mosaic_;
    cv::Mat mask_;
    cv::Mat drvxy_;
};

}

#endif