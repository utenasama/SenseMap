//
// Created by sensetime on 2020/12/1.
//

#ifndef SENSEMAP_POINTCLOUDALIGN_H
#define SENSEMAP_POINTCLOUDALIGN_H
#include "RGBDAlignUtility.h"
#include "RGBDPyramid.h"


class PointCloudAlign {
public:

    PointCloudAlign();
    PointCloudAlign(double d, int r, int s=1);

    void SetInput(const cv::Mat &src_depth, const Eigen::Vector4f src_K,
                  const cv::Mat &dst_depth, const Eigen::Vector4f dst_K);


    void SetSearchRadius(int r);

    void SetSampleStep(int s);

    void ComputeJacobian(Eigen::Matrix4d init, int &corres_num, float &residual,
                                       Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr);



    std::shared_ptr<XYZImage> GetSrcXYZ();
    std::shared_ptr<XYZImage> GetDstXYZ();

    void SaveDstCloud(std::string name, bool with_normal);

    cv::Mat src_depth_;
    cv::Mat dst_depth_;
    Eigen::Vector4f src_K_;
    Eigen::Vector4f dst_K_;

    std::shared_ptr<XYZImage> src_xyzs_;
    std::shared_ptr<XYZImage> dst_xyzs_;
    std::shared_ptr<XYZImage> dst_normals_;




    int search_radius_ = 5;
    int sample_step_ = 1;
    double max_dist_ = 0.05;

};

void ComputeDepthNormals(std::shared_ptr<XYZImage> &xyzs, std::shared_ptr<XYZImage> &normals, int step, bool full_neibour);


#endif //SENSEMAP_POINTCLOUDALIGN_H
