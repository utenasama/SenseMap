//
// Created by sensetime on 2020/12/1.
//

#include <fstream>
#include "PointCloudAlign.h"

PointCloudAlign::PointCloudAlign(){}
PointCloudAlign::PointCloudAlign(double d, int r, int s):
max_dist_(d), search_radius_(r), sample_step_(s){}

void PointCloudAlign::SetInput(const cv::Mat &src_depth, const Eigen::Vector4f src_K,
              const cv::Mat &dst_depth, const Eigen::Vector4f dst_K){
//    ? void CreateXYZImage(std::shared_ptr<XYZImage> &xyz, cv::Mat depth, Eigen::Vector4f K, int step);

    src_K_ = src_K;
    dst_K_ = dst_K;

    CreateXYZImage(src_xyzs_, src_depth, src_K, 1);
    CreateXYZImage(dst_xyzs_, dst_depth, dst_K, 1);
    ComputeDepthNormals(dst_xyzs_, dst_normals_, 1, true);
}

void PointCloudAlign::SetSearchRadius(int r){
    search_radius_ = r;
}

void PointCloudAlign::SetSampleStep(int s){
    sample_step_ = s;
}


std::shared_ptr<XYZImage> PointCloudAlign::GetSrcXYZ(){
    return src_xyzs_;
}
std::shared_ptr<XYZImage> PointCloudAlign::GetDstXYZ(){
    return dst_xyzs_;
}

void PointCloudAlign::SaveDstCloud(std::string name, bool with_normal){
    int width =dst_xyzs_->width_;
    int height = dst_xyzs_->height_;
    std::ofstream file(name);
    for(int i = 0; i<width; i++){
        for(int j = 0; j<height; j++){
            Eigen::Vector3f &dst_pt = (*dst_xyzs_)(i, j);
            Eigen::Vector3f &dst_n = (*dst_normals_)(i, j);

            if (std::isnan(dst_pt[2]) || dst_pt[2]<FLT_EPSILON) continue;
            if(with_normal)
                if(std::isnan(dst_n[2])) continue;
            file<<"v "<<dst_pt.transpose()<<std::endl;
            if(with_normal) file<<"vt "<<dst_n.transpose()<<std::endl;

        }
    }
    file.close();
}



void PointCloudAlign::ComputeJacobian(Eigen::Matrix4d init, int &corres_num, float &residual,
                     Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr){
    JtJ.setZero(); Jtr.setZero(); residual = 0; corres_num = 0;

    int width =src_xyzs_->width_;
    int height = src_xyzs_->height_;

    Eigen::Matrix3f R = init.block<3,3>(0,0).cast<float>();
    Eigen::Vector3f T = init.block<3,1>(0, 3).cast<float>();

    Eigen::Matrix3f dst_K;
    dst_K<<dst_K_(0), 0, dst_K_(2), 0, dst_K_(1), dst_K_(3), 0, 0, 1;

    for(int x=0; x<width; x+=sample_step_){
        for(int y= 0; y<height; y+=sample_step_){
            auto &src_pt = (*src_xyzs_)(x,y);
            if(std::isnan(src_pt[2]) || src_pt[2]<FLT_EPSILON) continue;

            Eigen::Vector3f trans_pt = R * src_pt + T;
            Eigen::Vector3f proj_pt = dst_K * trans_pt;
            int u = proj_pt[0]/proj_pt[2];
            int v = proj_pt[1]/proj_pt[2];

            const int umin = std::max(u - search_radius_, 0);
            const int vmin = std::max(v - search_radius_, 0);
            const int umax = std::min(u + search_radius_,  width - 1);
            const int vmax = std::min(v + search_radius_, height - 1);

            double min_dist = std::numeric_limits<double>::max();
            Eigen::Vector2i matched_uv(-1, -1);

            for (int u1 = umin; u1 <= umax; ++u1) {
                for (int v1 = vmin; v1 <= vmax; ++v1) {
                    Eigen::Vector3f &dst_pt = (*dst_xyzs_)(u1, v1);
                    Eigen::Vector3f &dst_n = (*dst_normals_)(u1, v1);

                    if (std::isnan(dst_pt[2]) || dst_pt[2]<FLT_EPSILON) continue;
                    if(std::isnan(dst_n[2])) continue;

                    double dist = (trans_pt - dst_pt).norm();
                    if (dist < min_dist) {
                        min_dist = dist;
                        matched_uv = Eigen::Vector2i(u1, v1);
                    }
                }
            }

//            std::cout<<"matched_uv: "<<matched_uv.transpose()<<"  min dist: "<<min_dist<<std::endl;

            if(matched_uv[0]==-1) continue;

            double scale = 1.0;
            const double lamda = 1.0;

            if(min_dist > 10 * max_dist_){
                continue;
            } else if(min_dist > max_dist_){
                scale = exp( lamda * (1 - std::fabs(min_dist)/max_dist_) );
            }
            Eigen::Vector3f &dst_pt = (*dst_xyzs_)(matched_uv[0], matched_uv[1]);
            Eigen::Vector3f &dst_n = (*dst_normals_)(matched_uv[0], matched_uv[1]);

            double re = scale * (trans_pt - dst_pt).transpose() * dst_n;

            residual += re * re;

            Eigen::Vector6d Jt;
            Jt.head<3>() = trans_pt.cross(dst_n).cast<double>();
            Jt.tail<3>() = dst_n.cast<double>();

            Jtr += scale * re * Jt;
            JtJ += scale * scale * Jt * Jt.transpose();
            corres_num++;
        }
    }
}

void
ComputeDepthNormals(std::shared_ptr<XYZImage> &xyzs, std::shared_ptr<XYZImage> &normals, int step, bool full_neibour) {

    normals = std::make_shared<XYZImage>(xyzs->width_, xyzs->height_);

    const int width = xyzs->width_;
    const int height = xyzs->height_;

    static std::vector<std::vector<int>> nn1 = {{-1, -1}, {0,  -1}, {1,  -1},
                                                    {1,  0},           {1,  1},
                                                    {0,  1}, {-1, 1}, {-1, 0}};

    static std::vector<std::vector<int>> nn2 = {{0,  -1}, {1,  0}, {0,  1}, {-1, 0}};

    std::vector<std::vector<int>> Neighbors;
    if (full_neibour) Neighbors = nn1;
    else Neighbors = nn2;
    int neibour_size = Neighbors.size();

    for (int x = 0; x < width; x += step) {
        for (int y = 0; y < height; y += step) {
            Eigen::Vector3f &pt = (*xyzs)(x,y);
            if (pt[2] < FLT_EPSILON || std::isnan(pt[2])) {
                continue;
            }

            Eigen::Vector3f N(0, 0, 0);
            int cnt = 0;
            for (int n = 0; n < neibour_size; n++) {
                Eigen::Vector2i np(x + Neighbors[n][0], y + Neighbors[n][1]);
                if (np[0] < 0 || np[0] >= width || np[1] < 0 || np[1] >= height) continue;
                auto NPt = (*xyzs)(np[0],np[1]);
                if (std::isnan(NPt[2]) || NPt[2] < FLT_EPSILON) continue;

                int n2 = (n + 1) % neibour_size;
                Eigen::Vector2i np2(x + Neighbors[n2][0], y + Neighbors[n2][1]);
                if (np2[0] < 0 || np2[0] >= width || np2[1] < 0 || np2[1] >= height) continue;
                auto NPt2 = (*xyzs)(np2[0],np2[1]);
                if (std::isnan(NPt2[2]) || NPt2[2] < FLT_EPSILON) continue;
                NPt2 -= pt;
                NPt -= pt;
                Eigen::Vector3f normal = NPt2.cross(NPt);
                N += normal.normalized();
                cnt++;
            }
            if (cnt == 0) continue;
            (*normals)(x, y) = N.normalized();
        }
    }
}
