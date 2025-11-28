#pragma once
#include <vector>
#include <Eigen/Eigen>

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <map>

namespace sensemap {

    class DepthMesh {
    public:
        DepthMesh()  {};
        ~DepthMesh() {};

    public:
        std::vector<Eigen::Vector3f> vertices_;
        std::vector<Eigen::Vector3i> faces_;
    };

    enum VertexFormat { POSITION = 3, POSITION_NORMAL = 6, POSITION_NORMAL_COLOR = 9};

    void GetMaxMeshSize(int &max_vtx_size, int &max_facet_size, int width, int height, int step = 1, VertexFormat vertex_format = POSITION_NORMAL);

    bool WarpFrameBuffer(cv::Mat& depth_buffer, 
                         const std::vector<Eigen::Vector3f> & vertices, 
                         const std::vector<Eigen::Vector3i> & faces, 
                         float min_depth = 0.0f, float max_depth = std::numeric_limits<float>::max(),
                         int cull_back_face = 0);

    bool WarpFrameBuffer(cv::Mat& depth_buffer_, 
                        const Eigen::Matrix3f& K, const Eigen::Matrix3f& R, const Eigen::Vector3f& t, 
                        DepthMesh* pGLObj);

    bool GenerateMesh(DepthMesh& pGLObj, 
                    const float *depth_buf, int stride, int width, int height,
                    const Eigen::Matrix3f &K, const Eigen::Matrix3f &R, const Eigen::Vector3f &T,
                    int step, float depth_thres, VertexFormat vertex_format,
                    float min_depth, float max_depth);


    bool WriteTriangleMeshObj(const std::string &filename, 
                              const DepthMesh &mesh);
}
