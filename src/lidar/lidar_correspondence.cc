//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "lidar_correspondence.h"
#include "base/pose.h"
#include "base/projection.h"
#include "util/logging.h"

#include <opencv2/opencv.hpp>
namespace sensemap {

void OutputCorrLidarPointCloud(const LidarSweep& sweep_ref, const LidarSweep& sweep_src,
                    const Eigen::Matrix4d& Tr_ref2src, 
                    const std::vector<struct LidarEdgeCorrespondence>& corner_corrs,
                    const std::vector<struct LidarPlaneCorrespondence>& surf_corrs,
                    std::string save_path = ""){
    std::string parent_save_path = GetParentDir(save_path);
    // std::cout << "OutputCorrLidarPointCloud GetParentDir: " << parent_save_path << ", " << save_path << std::endl;
    if (!parent_save_path.empty() && !boost::filesystem::exists(parent_save_path)){
        boost::filesystem::create_directories(parent_save_path);
    }
    Eigen::Matrix4d T_ref = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_src = Eigen::Matrix4d::Identity();
    // T_ref = Tr_ref2src;

    T_ref.topRows(3) = sweep_ref.InverseProjectionMatrix();
    T_src.topRows(3) = sweep_src.InverseProjectionMatrix();

    cv::RNG rng(12345);
    if (!corner_corrs.empty()){
        size_t num_corner = corner_corrs.size();
        std::vector<PlyPoint> ref_corner_ply;
        ref_corner_ply.reserve(num_corner);
        std::vector<PlyPoint> src_corner_ply;
        src_corner_ply.reserve(num_corner * 2);
        for (size_t i = 0; i < num_corner; i++){
            int r_color = rng.uniform(0, 255);
            int g_color = rng.uniform(0, 255);
            int b_color = rng.uniform(0, 255);
            
            PlyPoint a_pnt;
            const auto& a = corner_corrs.at(i).ref_point_;
            // const double value = corner_corrs.at(i).s;
            // uint8_t r_color = 255 * JetColormap::Red(value);
            // uint8_t g_color = 255 * JetColormap::Green(value);
            // uint8_t b_color = 255 * JetColormap::Blue(value);

            Eigen::Vector4d a_v4(a.x(), a.y(), a.z(), 1.0);
            // a_v4 = Tr_ref2src * a_v4;
            a_v4 = T_ref * a_v4;
            a_pnt.x = a_v4.x();
            a_pnt.y = a_v4.y();
            a_pnt.z = a_v4.z();
            a_pnt.r = r_color;
            a_pnt.g = g_color;
            a_pnt.b = b_color;
            ref_corner_ply.push_back(a_pnt);

            PlyPoint b1_pnt, b2_pnt;
            const auto& b1 = corner_corrs.at(i).src_point_a_;
            Eigen::Vector4d b1_v4(b1.x(), b1.y(), b1.z(), 1.0);
            b1_v4 = T_src * b1_v4;
            b1_pnt.x = b1_v4.x();
            b1_pnt.y = b1_v4.y();
            b1_pnt.z = b1_v4.z();
            b1_pnt.r = r_color;
            b1_pnt.g = g_color;
            b1_pnt.b = b_color;
            src_corner_ply.push_back(b1_pnt);
            const auto& b2 = corner_corrs.at(i).src_point_b_;
            Eigen::Vector4d b2_v4(b2.x(), b2.y(), b2.z(), 1.0);
            b2_v4 = T_src * b2_v4;
            b2_pnt.x = b2_v4.x();
            b2_pnt.y = b2_v4.y();
            b2_pnt.z = b2_v4.z();
            b2_pnt.r = r_color;
            b2_pnt.g = g_color;
            b2_pnt.b = b_color;
            src_corner_ply.push_back(b2_pnt);
        }
        WriteBinaryPlyPoints(save_path + "_corner_ref.ply", ref_corner_ply, false, true);
        WriteBinaryPlyPoints(save_path + "_corner_src.ply", src_corner_ply, false, true);
    }

    if (!surf_corrs.empty()){
        size_t num_surf = surf_corrs.size();
        std::vector<PlyPoint> ref_surf_ply;
        ref_surf_ply.reserve(num_surf);
        std::vector<PlyPoint> src_surf_ply;
        src_surf_ply.reserve(num_surf * 3);
        for (size_t i = 0; i < num_surf; i++){
            int r_color = rng.uniform(0, 255);
            int g_color = rng.uniform(0, 255);
            int b_color = rng.uniform(0, 255);

            PlyPoint a_pnt;
            const auto& a = surf_corrs.at(i).ref_point_;
            // const double value = surf_corrs.at(i).s;
            // uint8_t r_color = 255 * JetColormap::Red(value);
            // uint8_t g_color = 255 * JetColormap::Green(value);
            // uint8_t b_color = 255 * JetColormap::Blue(value);

            Eigen::Vector4d a_v4(a.x(), a.y(), a.z(), 1.0);
            a_v4 = T_ref * a_v4;
            a_pnt.x = a_v4.x();
            a_pnt.y = a_v4.y();
            a_pnt.z = a_v4.z();
            a_pnt.r = r_color;
            a_pnt.g = g_color;
            a_pnt.b = b_color;
            ref_surf_ply.push_back(a_pnt);

            PlyPoint b1_pnt, b2_pnt, b3_pnt;
            const auto& b1 = surf_corrs.at(i).src_point_j_;
            Eigen::Vector4d b1_v4(b1.x(), b1.y(), b1.z(), 1.0);
            b1_v4 = T_src * b1_v4;
            b1_pnt.x = b1_v4.x();
            b1_pnt.y = b1_v4.y();
            b1_pnt.z = b1_v4.z();
            b1_pnt.r = r_color;
            b1_pnt.g = g_color;
            b1_pnt.b = b_color;
            src_surf_ply.push_back(b1_pnt);
            const auto& b2 = surf_corrs.at(i).src_point_l_;
            Eigen::Vector4d b2_v4(b2.x(), b2.y(), b2.z(), 1.0);
            b2_v4 = T_src * b2_v4;
            b2_pnt.x = b2_v4.x();
            b2_pnt.y = b2_v4.y();
            b2_pnt.z = b2_v4.z();
            b2_pnt.r = r_color;
            b2_pnt.g = g_color;
            b2_pnt.b = b_color;
            src_surf_ply.push_back(b2_pnt);
            const auto& b3 = surf_corrs.at(i).src_point_m_;
            Eigen::Vector4d b3_v4(b3.x(), b3.y(), b3.z(), 1.0);
            b3_v4 = T_src * b3_v4;
            b3_pnt.x = b3_v4.x();
            b3_pnt.y = b3_v4.y();
            b3_pnt.z = b3_v4.z();
            b3_pnt.r = r_color;
            b3_pnt.g = g_color;
            b3_pnt.b = b_color;
            src_surf_ply.push_back(b3_pnt);
        }
        WriteBinaryPlyPoints(save_path + "_surf_ref.ply", ref_surf_ply, false, true);
        WriteBinaryPlyPoints(save_path + "_surf_src.ply", src_surf_ply, false, true);
    }

    {
        LidarPointCloud ref_less_features, ref_less_features_t; 
        LidarPointCloud ref_less_surfs = sweep_ref.GetSurfPointsLessFlat();
        LidarPointCloud ref_less_corners = sweep_ref.GetCornerPointsLessSharp();
        ref_less_features = ref_less_surfs;
        ref_less_features += ref_less_corners;

        LidarPointCloud src_less_features, src_less_features_t; 
        LidarPointCloud src_less_surfs = sweep_src.GetSurfPointsLessFlat();
        LidarPointCloud src_less_corners = sweep_src.GetCornerPointsLessSharp();
        // LidarPointCloud src_surfs = sweep_src.GetSurfPointsFlat();
        src_less_features = src_less_surfs;
        src_less_features += src_less_corners;

        LidarPointCloud::TransfromPlyPointCloud (ref_less_features, ref_less_features_t, T_ref);
        WriteBinaryPlyPoints(save_path + "_ref_pnts.ply", ref_less_features_t.Convert2Ply(), false, true);

        LidarPointCloud::TransfromPlyPointCloud (src_less_features, src_less_features_t, T_src);
        WriteBinaryPlyPoints(save_path + "_src_pnts.ply", src_less_features_t.Convert2Ply(), false, true);

        // LidarPointCloud::TransfromPlyPointCloud (ref_less_features, ref_less_features_t, T_ref);
        // WriteBinaryPlyPoints(save_path + "_ref2world_pnts.ply", ref_less_features_t.Convert2Ply(), false, true);

        // LidarPointCloud::TransfromPlyPointCloud (src_less_features, src_less_features_t, T_src);
        // WriteBinaryPlyPoints(save_path + "_src2world_pnts.ply", src_less_features_t.Convert2Ply(), false, true);
    }
    return;
}

void TransformSweepToSweep(Eigen::Matrix4d& Tr, LidarPoint const *const pi, 
                           LidarPoint *const po, double& s) {
    //interpolation ratio
    // double s;
    if (DISTORTION)
        s = (pi->scanid - int(pi->scanid)) / SCAN_PERIOD;
    else
        s = 1.0;
    //s = 1;
    Eigen::Quaterniond q_last_curr(Tr.topLeftCorner<3,3>());
    Eigen::Vector3d t_last_curr = Tr.topRightCorner<3,1>();
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
    Eigen::Vector3d t_point_last = s * t_last_curr;

    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    Eigen::Vector3d normal(pi->nx, pi->ny, pi->nz);
    Eigen::Vector3d un_normal=q_point_last * normal;

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->nx = un_normal.x();
    po->ny = un_normal.y();
    po->nz = un_normal.z();
    po->scanid = pi->scanid;
}

bool LidarPointsCorrespondence(LidarSweep& sweep_ref, LidarSweep& sweep_src,
                    Eigen::Matrix4d& Tr_ref2src, 
                    std::vector<struct LidarEdgeCorrespondence>& corner_corrs,
                    std::vector<struct LidarPlaneCorrespondence>& surf_corrs,
                    std::vector<struct LidarPointCorrespondence>& pnt_corrs,
                    std::string save_path){
    // int corner_num = 0;
    int corner_num = 
            CornerCorrespondence(sweep_ref, sweep_src, Tr_ref2src, corner_corrs);
    int surf_num =
            SurfCorrespondence(sweep_ref, sweep_src, Tr_ref2src, surf_corrs);
    int pnt_num = 0;
            // PntCorrespondence(sweep_ref, sweep_src, Tr_ref2src, pnt_corrs);

    // OutputCorrLidarPointCloud(sweep_ref, sweep_src, Tr_ref2src, corner_corrs, surf_corrs, save_path);

    std::cout << "sweep pair (" << sweep_ref.SweepID() << " & " 
            << sweep_src.SweepID() << " : " << corner_num << "+" << surf_num << "+" << pnt_num << " pairs)  ";
    if (surf_num < 15 && corner_num + surf_num < 100){
        printf("less correspondence!\n");
        return false;
    }

    return true;
}

// bool LidarPointsCorrespondence(LidarSweep& sweep_ref, 
//                 Eigen::Matrix4d& Tr_r2w, 
//                 Reconstruction *reconstruction,
//                 kd_tree_t & corner_kdtree,
//                 kd_tree_t & surf_kdtree,
//                 std::unordered_map<size_t, std::vector<size_t>> & corner_corrs,
//                 std::unordered_map<size_t, std::vector<size_t>> & surf_corrs){
//     LidarPointCloud global_corner_points = reconstruction->CornerMap();
//     int corner_num = 
//             CornerCorrespondence(sweep_ref, Tr_r2w, global_corner_points, corner_kdtree, corner_corrs);

//     LidarPointCloud global_surf_points = reconstruction->SurfaceMap();
//     int surf_num =
//             SurfCorrespondence(sweep_ref, Tr_r2w, global_surf_points, surf_kdtree, surf_corrs);

//     // OutputCorrLidarPointCloud(sweep_ref, sweep_src, Tr_ref2src, corner_corrs, surf_corrs);
//     // std::cout << "sweep pair (" << sweep_ref.SweepID() << " : " << corner_num << "+" << surf_num << " pairs)   ";
//     if (corner_num + surf_num < 60){
//         std::cout  << "sweep pair (" << sweep_ref.SweepID() << " : " << corner_num << "+" << surf_num << " pairs)   less correspondence!" << std::endl;
//         return false;
//     }

//     // std::cout << "lidar correspondence size: " << corner_num << "&" << surf_num << std::endl;

//     return true;
// }

int CornerCorrespondence(LidarSweep& sweep_ref, LidarSweep& sweep_src,
                    Eigen::Matrix4d& Tr_ref2src, 
                    std::vector<struct LidarEdgeCorrespondence>& corner_corrs){
    const LidarPointCloud& ref_corners = sweep_ref.GetCornerPointsSharp();
    int corner_points_num = ref_corners.points.size();

    // src sweep point cloud build kdtree
    const LidarPointCloud& src_less_corners = sweep_src.GetCornerPointsLessSharp();
    kd_tree_t corners_index(3 /*dim*/, src_less_corners, 
                            nanoflann::KDTreeSingleIndexAdaptorParams(10) );
	corners_index.buildIndex();

    LidarPoint pointSel;
    size_t  pointSearchInd;
    float pointSearchSqDis;

    int corner_correspondence = 0;
    double sum_dist = 0;
    const size_t num_results = 1; 
    for (int i = 0; i < corner_points_num; ++i) {
        double s = 1.0;
        
        Eigen::Vector3d ref_point(ref_corners.points[i].x,
                                    ref_corners.points[i].y,
                                    ref_corners.points[i].z);
        TransformSweepToSweep(Tr_ref2src, &(ref_corners.points[i]), &pointSel, s);
        
        float query_pt[3] = {pointSel.x, pointSel.y, pointSel.z};
		nanoflann::KNNResultSet<float> resultSet(num_results);
		resultSet.init(&pointSearchInd, &pointSearchSqDis );
		corners_index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

        // normal correspondence
        Eigen::Vector3d normal_curr(pointSel.nx, pointSel.ny, pointSel.nz);
        Eigen::Vector3d normal_last(src_less_corners.points[pointSearchInd].nx, 
                                    src_less_corners.points[pointSearchInd].ny, 
                                    src_less_corners.points[pointSearchInd].nz);
        if (normal_curr.norm() > 0.5 && normal_last.norm() > 0.5 && normal_curr.dot(normal_last) < -1e-4){
            continue;
        }

        int closestPointInd = -1, minPointInd2 = -1;
        float dist_threshold = DISTANCE_SQ_THRESHOLD + 0.3 * ref_point.norm();
        if (pointSearchSqDis < dist_threshold)
        {
            closestPointInd = pointSearchInd;
            int closestPointScanID = int(src_less_corners.points[closestPointInd].scanid);

            double minPointSqDis2 = dist_threshold;
            // search in the direction of increasing scan line
            for (int j = closestPointInd + 1; j < (int)src_less_corners.points.size(); ++j)
            {
                // if in the same scan line, continue
                if (int(src_less_corners.points[j].scanid) <= closestPointScanID)
                    continue;

                // if not in nearby scans, end the loop
                if (int(src_less_corners.points[j].scanid) > (closestPointScanID + NEARBY_SCAN))
                    break;

                double pointSqDis = (src_less_corners.points[j].x - pointSel.x) *
                                        (src_less_corners.points[j].x - pointSel.x) +
                                    (src_less_corners.points[j].y - pointSel.y) *
                                        (src_less_corners.points[j].y - pointSel.y) +
                                    (src_less_corners.points[j].z - pointSel.z) *
                                        (src_less_corners.points[j].z - pointSel.z);

                if (pointSqDis < minPointSqDis2)
                {
                    // find nearer point
                    minPointSqDis2 = pointSqDis;
                    minPointInd2 = j;
                }
            }

            // search in the direction of decreasing scan line
            for (int j = closestPointInd - 1; j >= 0; --j)
            {
                // if in the same scan line, continue
                if (int(src_less_corners.points[j].scanid) >= closestPointScanID)
                    continue;

                // if not in nearby scans, end the loop
                if (int(src_less_corners.points[j].scanid) < (closestPointScanID - NEARBY_SCAN))
                    break;

                double pointSqDis = (src_less_corners.points[j].x - pointSel.x) *
                                        (src_less_corners.points[j].x - pointSel.x) +
                                    (src_less_corners.points[j].y - pointSel.y) *
                                        (src_less_corners.points[j].y - pointSel.y) +
                                    (src_less_corners.points[j].z - pointSel.z) *
                                        (src_less_corners.points[j].z - pointSel.z);

                if (pointSqDis < minPointSqDis2)
                {
                    // find nearer point
                    minPointSqDis2 = pointSqDis;
                    minPointInd2 = j;
                }
            }
        }
        if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
        {
            Eigen::Vector3d curr_point(ref_corners.points[i].x,
                                       ref_corners.points[i].y,
                                       ref_corners.points[i].z);
            Eigen::Vector3d last_point_a(src_less_corners.points[closestPointInd].x,
                                        src_less_corners.points[closestPointInd].y,
                                        src_less_corners.points[closestPointInd].z);
            Eigen::Vector3d last_point_b(src_less_corners.points[minPointInd2].x,
                                        src_less_corners.points[minPointInd2].y,
                                        src_less_corners.points[minPointInd2].z);
            
            Eigen::Vector3d ref_trans_pnts(pointSel.x, pointSel.y, pointSel.z);

            Eigen::Vector3d nu = (ref_trans_pnts - last_point_a).cross(ref_trans_pnts - last_point_b);
            double de = (last_point_a - last_point_b).norm();
            double dist = nu.norm() / (de);
            sum_dist += dist;

            float max_intensity = 70;
            double intensity_ref = 1.0 - std::exp(-ref_corners.points[i].intensity * ref_corners.points[i].intensity / max_intensity);
            double intensity_a = 1.0 - std::exp(-src_less_corners.points[closestPointInd].intensity * src_less_corners.points[closestPointInd].intensity / max_intensity);
            double intensity_b = 1.0 - std::exp(-src_less_corners.points[minPointInd2].intensity * src_less_corners.points[minPointInd2].intensity / max_intensity);
            // double intensity_factor = intensity_ref * std::min(intensity_a, intensity_b);
            double intensity_factor = intensity_ref * (intensity_a + intensity_b) / 2;
            struct LidarEdgeCorrespondence corner_corr = {curr_point, last_point_a,
                                                          last_point_b, i, 
                                                          closestPointInd,
                                                          minPointInd2, intensity_factor, dist};
            // struct LidarEdgeCorrespondence corner_corr = {curr_point, last_point_a,
            //                                               last_point_b, s};            
            corner_corrs.push_back(corner_corr);
            corner_correspondence++;
        }
    }


    // double mean_dist = sum_dist / (double)corner_correspondence;
    // std::cout << "ori corner_correspondence: " << corner_correspondence << std::endl;
    // corner_correspondence = 0;
    // for (int i = 0; i < corner_corrs.size(); i++){
    //     if (corner_corrs.at(i).dist > 1.2 * mean_dist + 0.2){
    //         continue;
    //     }
    //     corner_corrs[corner_correspondence] = corner_corrs[i];
    //     corner_correspondence++;
    // }
    // corner_corrs.resize(corner_correspondence);

    // std::cout << "new corner_correspondence: " << corner_correspondence << std::endl;

    return corner_correspondence;
}


// int CornerCorrespondence(LidarSweep& sweep_ref, 
//                     Eigen::Matrix4d& Tr_ref2w, 
//                     LidarPointCloud& global_points,
//                     kd_tree_t & corner_kdtree,
//                     std::unordered_map<size_t, std::vector<size_t>> & corner_corrs){
//     if (!corner_corrs.empty()){
//         corner_corrs.clear();
//     }
//     LidarPointCloud ref_corners = sweep_ref.GetCornerPointsSharp();
//     int corner_points_num = ref_corners.points.size();

//     LidarPoint pointSel;
//     const size_t num_results = 5;
//     // // test
//     // LidarPoint test_ref_cor;
//     // LidarPoint test_map_cor;
//     // std::cout << global_points.points.size() << std::endl;

//     for (size_t i = 0; i < corner_points_num; ++i) {
//         double s = 1.0;
//         TransformSweepToSweep(Tr_ref2w, &(ref_corners.points[i]), &pointSel, s);
//         float query_pt[3] = {pointSel.x, pointSel.y, pointSel.z};
        
//         size_t num_knn = num_results * 2;
//         std::vector<size_t> ret_index_t(num_knn);
//         std::vector<float> out_dist_sqr_t(num_knn);
//         while(1){
//             std::vector<size_t> ret_index(num_knn);
//             std::vector<float> out_dist_sqr(num_knn);
//             corner_kdtree.knnSearch(&query_pt[0], num_knn, &ret_index[0], &out_dist_sqr[0]);

//             ret_index_t.resize(num_knn);
//             out_dist_sqr_t.resize(num_knn);

//             int k = 0;
//             // std::cout << num_knn << " id: ";
//             for (int j = 0; j < num_knn; j++){
//                 // std::cout << global_points.points.at((int)ret_index[j]).lidar_id[0] << "&" << sweep_ref.SweepID() << " ";
//                 if (global_points.points.at((int)ret_index[j]).lidar_id[0]
//                     == sweep_ref.SweepID()){
//                     continue;
//                 }
//                 ret_index_t[k] = ret_index[j];
//                 out_dist_sqr_t[k] = out_dist_sqr[j];
//                 k++;
//             }
//             // std::cout << std::endl;

//             if (k >= num_results){
//                 break;
//             }
//             num_knn *= 2;
//         }
        
//         ret_index_t.resize(num_results);
//         out_dist_sqr_t.resize(num_results);

//         if(out_dist_sqr_t[num_results - 1] < DISTANCE_SQ_THRESHOLD){
//             corner_corrs.insert(std::make_pair((size_t)i, ret_index_t));
//         }
//     }

//     return corner_corrs.size();
// }


int SurfCorrespondence(LidarSweep& sweep_ref, LidarSweep& sweep_src,
                       Eigen::Matrix4d& Tr_ref2src, 
                       std::vector<struct LidarPlaneCorrespondence>& surf_corrs){
    LidarPointCloud ref_surfs = sweep_ref.GetSurfPointsFlat();
    int surf_points_num = ref_surfs.points.size();

    // src sweep point cloud build kdtree
    LidarPointCloud src_less_surfs = sweep_src.GetSurfPointsLessFlat();
    kd_tree_t surfs_index(3 /*dim*/, src_less_surfs, 
                          nanoflann::KDTreeSingleIndexAdaptorParams(10) );
	surfs_index.buildIndex();

    LidarPoint pointSel;
    size_t  pointSearchInd;
    float pointSearchSqDis;

    int plane_correspondence = 0;
    double sum_dist = 0;
    const size_t num_results = 1;
    for (int i = 0; i < surf_points_num; ++i) {
        double s = 1.0;
        
        Eigen::Vector3d ref_point(ref_surfs.points[i].x,
                                    ref_surfs.points[i].y,
                                    ref_surfs.points[i].z);
        TransformSweepToSweep(Tr_ref2src, &(ref_surfs.points[i]), &pointSel, s);

        float query_pt[3] = {pointSel.x, pointSel.y, pointSel.z};
		nanoflann::KNNResultSet<float> resultSet(num_results);
		resultSet.init(&pointSearchInd, &pointSearchSqDis );
		surfs_index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

        // normal correspondence
        Eigen::Vector3d normal_curr(pointSel.nx, pointSel.ny, pointSel.nz);
        Eigen::Vector3d normal_last(src_less_surfs.points[pointSearchInd].nx, 
                                    src_less_surfs.points[pointSearchInd].ny, 
                                    src_less_surfs.points[pointSearchInd].nz);
        if (normal_curr.norm() > 0.5 && normal_last.norm() > 0.5 && normal_curr.dot(normal_last) < -1e-4){
            continue;
        }

        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
        float dist_threshold = DISTANCE_SQ_THRESHOLD + 0.3 * ref_point.norm();
        if (pointSearchSqDis < dist_threshold) {
            closestPointInd = pointSearchInd;

            // get closest point's scan ID
            int closestPointScanID = int(src_less_surfs.points[closestPointInd].scanid);
            double minPointSqDis2 = dist_threshold;
            double minPointSqDis3 = dist_threshold;

            // search in the direction of increasing scan line
            for (int j = closestPointInd + 1; j < (int)src_less_surfs.points.size(); ++j)
            {
                // if not in nearby scans, end the loop
                if (int(src_less_surfs.points[j].scanid) > 
                    (closestPointScanID + NEARBY_SCAN))
                    break;

                double pointSqDis = (src_less_surfs.points[j].x - pointSel.x) *
                                    (src_less_surfs.points[j].x - pointSel.x) +
                                    (src_less_surfs.points[j].y - pointSel.y) *
                                    (src_less_surfs.points[j].y - pointSel.y) +
                                    (src_less_surfs.points[j].z - pointSel.z) *
                                    (src_less_surfs.points[j].z - pointSel.z);

                // if in the same or lower scan line
                if (int(src_less_surfs.points[j].scanid) <= closestPointScanID
                    && pointSqDis < minPointSqDis2) {
                    minPointSqDis2 = pointSqDis;
                    minPointInd2 = j;
                }
                // if in the higher scan line
                else if (int(src_less_surfs.points[j].scanid) > 
                         closestPointScanID && pointSqDis < minPointSqDis3) {
                    minPointSqDis3 = pointSqDis;
                    minPointInd3 = j;
                }
            }

            // search in the direction of decreasing scan line
            for (int j = closestPointInd - 1; j >= 0; --j)
            {
                // if not in nearby scans, end the loop
                if (int(src_less_surfs.points[j].scanid) 
                    < (closestPointScanID - NEARBY_SCAN))
                    break;

                double pointSqDis = (src_less_surfs.points[j].x - pointSel.x) *
                                    (src_less_surfs.points[j].x - pointSel.x) +
                                    (src_less_surfs.points[j].y - pointSel.y) *
                                    (src_less_surfs.points[j].y - pointSel.y) +
                                    (src_less_surfs.points[j].z - pointSel.z) *
                                    (src_less_surfs.points[j].z - pointSel.z);

                // if in the same or higher scan line
                if (int(src_less_surfs.points[j].scanid) >= closestPointScanID
                     && pointSqDis < minPointSqDis2){
                    minPointSqDis2 = pointSqDis;
                    minPointInd2 = j;
                }
                else if (int(src_less_surfs.points[j].scanid) 
                        < closestPointScanID && pointSqDis < minPointSqDis3){
                    // find nearer point
                    minPointSqDis3 = pointSqDis;
                    minPointInd3 = j;
                }
            }

            if (minPointInd2 >= 0 && minPointInd3 >= 0){

                Eigen::Vector3d curr_point(ref_surfs.points[i].x,
                                           ref_surfs.points[i].y,
                                           ref_surfs.points[i].z);
                Eigen::Vector3d last_point_a(
                                    src_less_surfs.points[closestPointInd].x,
                                    src_less_surfs.points[closestPointInd].y,
                                    src_less_surfs.points[closestPointInd].z);
                Eigen::Vector3d last_point_b(
                                        src_less_surfs.points[minPointInd2].x,
                                        src_less_surfs.points[minPointInd2].y,
                                        src_less_surfs.points[minPointInd2].z);
                Eigen::Vector3d last_point_c(
                                        src_less_surfs.points[minPointInd3].x,
                                        src_less_surfs.points[minPointInd3].y,
                                        src_less_surfs.points[minPointInd3].z);

                // struct LidarPlaneCorrespondence surf_corr = {curr_point, 
                //                                             last_point_a,
                //                                             last_point_b, 
                //                                             last_point_c,
                //                                             s};            

                Eigen::Vector3d ref_trans_pnts(pointSel.x, pointSel.y, pointSel.z);
                Eigen::Vector3d plane_norm = (last_point_a - last_point_b).cross(last_point_a - last_point_c);
                plane_norm.normalize();
                double dist = std::abs((ref_trans_pnts - last_point_a).dot(plane_norm));
                sum_dist += dist;

                float max_intensity = 70;
                double intensity_ref = 1.0 - std::exp(-ref_surfs.points[i].intensity * ref_surfs.points[i].intensity / max_intensity);
                double intensity_a = 1.0 - std::exp(-src_less_surfs.points[closestPointInd].intensity * src_less_surfs.points[closestPointInd].intensity / max_intensity);
                double intensity_b = 1.0 - std::exp(-src_less_surfs.points[minPointInd2].intensity * src_less_surfs.points[minPointInd2].intensity / max_intensity);
                double intensity_c = 1.0 - std::exp(-src_less_surfs.points[minPointInd3].intensity * src_less_surfs.points[minPointInd3].intensity / max_intensity);
                // double intensity_factor = intensity_ref * std::min(std::min(intensity_a, intensity_b), intensity_c);
                double intensity_factor = intensity_ref * (intensity_a + intensity_b + intensity_c) / 3.0 ;
                // std::cout << "intensity_ref: " << intensity_ref << std::endl;
                struct LidarPlaneCorrespondence surf_corr = {curr_point, 
                                                            last_point_a,
                                                            last_point_b, 
                                                            last_point_c,
                                                            i, 
                                                            closestPointInd,
                                                            minPointInd2,
                                                            minPointInd3,
                                                            intensity_factor, dist};         
                surf_corrs.push_back(surf_corr);
                
                plane_correspondence++;
            }
        }
    }

    // // std::cout << "ori plane_correspondence: " << plane_correspondence << std::endl;
    // double mean_dist = sum_dist / (double)plane_correspondence;
    // plane_correspondence = 0;
    // for (int i = 0; i < surf_corrs.size(); i++){
    //     if (surf_corrs.at(i).dist > 1.2 * mean_dist + 0.2){
    //         continue;
    //     }
    //     surf_corrs.at(plane_correspondence) = surf_corrs.at(i);
    //     plane_correspondence++;
    // }
    // surf_corrs.resize(plane_correspondence);

    // std::cout << "new plane_correspondence: " << plane_correspondence << std::endl;
    return plane_correspondence;
}

// int SurfCorrespondence(LidarSweep& sweep_ref, 
//                        Eigen::Matrix4d& Tr_ref2w, 
//                        LidarPointCloud& global_points,
//                        kd_tree_t & surf_kdtree, 
//                        std::unordered_map<size_t, std::vector<size_t>> & surf_corrs){
//     if (!surf_corrs.empty()){
//         surf_corrs.clear();
//     }
//     LidarPointCloud ref_corners = sweep_ref.GetSurfPointsFlat();
//     int corner_points_num = ref_corners.points.size();

//     LidarPoint pointSel;
//     const size_t num_results = 5;
//     size_t ret_index;
//     float out_dist_sqr;
//     for (size_t i = 0; i < corner_points_num; ++i) {
//         double s = 1.0;
//         TransformSweepToSweep(Tr_ref2w, &(ref_corners.points[i]), &pointSel, s);
//         float query_pt[3] = {pointSel.x, pointSel.y, pointSel.z};

//         size_t num_knn = num_results * 2;
//         std::vector<size_t> ret_index_t(num_knn);
//         std::vector<float> out_dist_sqr_t(num_knn);
//         while(1){
//             std::vector<size_t> ret_index(num_knn);
//             std::vector<float> out_dist_sqr(num_knn);
//             surf_kdtree.knnSearch(&query_pt[0], num_knn, &ret_index[0], &out_dist_sqr[0]);

//             ret_index_t.resize(num_knn);
//             out_dist_sqr_t.resize(num_knn);

//             int k = 0;
//             for (int j = 0; j < num_knn; j++){
//                 if (global_points.points.at((int)ret_index[j]).lidar_id[0]
//                     == sweep_ref.SweepID()){
//                     continue;
//                 }
//                 ret_index_t[k] = ret_index[j];
//                 out_dist_sqr_t[k] = out_dist_sqr[j];
//                 k++;
//             }

//             if (k >= num_results){
//                 break;
//             }
//             num_knn *= 2;
//         }

//         ret_index_t.resize(num_results);
//         out_dist_sqr_t.resize(num_results);

//         if(out_dist_sqr_t[num_results - 1] < DISTANCE_SQ_THRESHOLD){
//             surf_corrs.insert(std::make_pair((size_t)i, ret_index_t));
//         }
//     }

//     return surf_corrs.size();
// }

int PntCorrespondence(LidarSweep& sweep_ref, LidarSweep& sweep_src,
                    Eigen::Matrix4d& Tr_ref2src, 
                    std::vector<struct LidarPointCorrespondence>& pnt_corrs){
    const LidarPointCloud& ref_cor_pnts = sweep_ref.GetCornerPointsLessSharp();
    // const LidarPointCloud& ref_surf_pnts = sweep_ref.GetSurfPointsFlat();
    LidarPointCloud ref_pnts = ref_cor_pnts;
    // ref_pnts += ref_surf_pnts;
    int points_num = ref_pnts.points.size();

    // src sweep point cloud build kdtree
    const LidarPointCloud& src_cor_pnts = sweep_src.GetCornerPointsLessSharp ();
    const LidarPointCloud& src_surf_pnts = sweep_src.GetSurfPointsLessFlat();
    LidarPointCloud src_less_pnts = src_cor_pnts;
    src_less_pnts += src_surf_pnts;
    kd_tree_t pnts_index(3 /*dim*/, src_less_pnts, 
                            nanoflann::KDTreeSingleIndexAdaptorParams(10) );
	pnts_index.buildIndex();

    LidarPoint pointSel;
    size_t  pointSearchInd;
    float pointSearchSqDis;

    int pnt_correspondence = 0;
    double sum_dist = 0;
    const size_t num_results = 1; 
    for (int i = 0; i < points_num; ++i) {
        double s = 1.0;

        TransformSweepToSweep(Tr_ref2src, &(ref_pnts.points[i]), &pointSel, s);
        
        float query_pt[3] = {pointSel.x, pointSel.y, pointSel.z};
		nanoflann::KNNResultSet<float> resultSet(num_results);
		resultSet.init(&pointSearchInd, &pointSearchSqDis );
		pnts_index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

        // normal correspondence
        Eigen::Vector3d normal_curr(pointSel.nx, pointSel.ny, pointSel.nz);
        Eigen::Vector3d normal_last(src_less_pnts.points[pointSearchInd].nx, 
                                    src_less_pnts.points[pointSearchInd].ny, 
                                    src_less_pnts.points[pointSearchInd].nz);
        if (normal_curr.norm() > 0.5 && normal_last.norm() > 0.5 && normal_curr.dot(normal_last) < -1e-4){
            continue;
        }

        if (pointSearchSqDis > DISTANCE_SQ_THRESHOLD){
            continue;
        }

        Eigen::Vector3d curr_point(ref_pnts.points[i].x,
                                    ref_pnts.points[i].y,
                                    ref_pnts.points[i].z);

        Eigen::Vector3d last_point_a(src_less_pnts.points[pointSearchInd].x,
                                        src_less_pnts.points[pointSearchInd].y,
                                        src_less_pnts.points[pointSearchInd].z);
        
        Eigen::Vector3d ref_trans_pnts(pointSel.x, pointSel.y, pointSel.z);
        double dist = (ref_trans_pnts - last_point_a).norm() / curr_point.norm();
        sum_dist += dist;

        float max_intensity = 70;
        double intensity_ref = 1.0 - std::exp(-ref_pnts.points[i].intensity * ref_pnts.points[i].intensity / max_intensity);
        double intensity_src = 1.0 - std::exp(-src_less_pnts.points[pointSearchInd].intensity * src_less_pnts.points[pointSearchInd].intensity / max_intensity);
        double intensity_factor = intensity_ref * intensity_src;

        struct LidarPointCorrespondence pnt_corr = {curr_point, last_point_a, i, 
                                                    (int)pointSearchInd, intensity_factor, dist};
        
        pnt_corrs.push_back(pnt_corr);
        pnt_correspondence++;
    }


    double mean_dist = sum_dist / (double)pnt_correspondence;
    // std::cout << "ori pnt_correspondence: " << pnt_correspondence << std::endl;
    pnt_correspondence = 0;
    for (int i = 0; i < pnt_corrs.size(); i++){
        if (pnt_corrs.at(i).dist > 1.2 * mean_dist + 0.2){
            continue;
        }
        pnt_corrs[pnt_correspondence] = pnt_corrs[i];
        pnt_correspondence++;
    }
    pnt_corrs.resize(pnt_correspondence);

    // std::cout << "new pnt_correspondence: " << pnt_correspondence << std::endl;

    return pnt_correspondence;
}

double LidarPointsEdgeResidual(const Eigen::Vector3d curr_point_t, 
                         const Eigen::Vector3d last_point_a_t,
					     const Eigen::Vector3d last_point_b_t, 
                         double q1[4], double t1[3], double q2[4], double t2[3],
                         Eigen::Matrix4d& Tr_ref2src, Eigen::Matrix4d& Tr_lidar2camera){
    typedef double T;

    Eigen::Vector3d curr_point(curr_point_t); 
    Eigen::Vector3d last_point_a(last_point_a_t); 
    Eigen::Vector3d last_point_b(last_point_b_t);

    Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
    Eigen::Matrix<T, 3, 1> spa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
    Eigen::Matrix<T, 3, 1> spb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

    //Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
    Eigen::Quaternion<T> q_ref{q1[0], q1[1], q1[2], q1[3]};
    Eigen::Matrix<T, 3, 1> t_ref{t1[0], t1[1], t1[2]};
    Eigen::Quaternion<T> q_src{q2[0], q2[1], q2[2], q2[3]};
    Eigen::Matrix<T, 3, 1> t_src{t2[0], t2[1], t2[2]};

    Eigen::Matrix<T, 3, 1> lp, lpa, lpb;
    lp = q_ref.inverse() * cp - q_ref.inverse() * t_ref;
    lpa = q_src.inverse() * spa - q_src.inverse() * t_src;
    lpb = q_src.inverse() * spb - q_src.inverse() * t_src;

    Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
    Eigen::Matrix<T, 3, 1> de = lpa - lpb;

    T residual[3];
    residual[0] = nu.x() / de.norm();
    residual[1] = nu.y() / de.norm();
    residual[2] = nu.z() / de.norm();

    return  std::sqrt(residual[0]*residual[0] + residual[1]*residual[1] + residual[2]*residual[2]);
}


double LidarPointsPlaneResidual(const Eigen::Vector3d curr_point_t, 
                                const Eigen::Vector3d last_point_j_t,
                                const Eigen::Vector3d last_point_l_t, 
                                const Eigen::Vector3d last_point_m_t,
                                double q1[4], double t1[3], 
                                double q2[4], double t2[3],
                                Eigen::Matrix4d& Tr_ref2src, 
                                Eigen::Matrix4d& Tr_lidar2camera){
    typedef double T;

    Eigen::Vector3d curr_point(curr_point_t); 
    Eigen::Vector3d last_point_j(last_point_j_t); 
    Eigen::Vector3d last_point_l(last_point_l_t);
    Eigen::Vector3d last_point_m(last_point_m_t);

    Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
    Eigen::Matrix<T, 3, 1> spj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
    Eigen::Matrix<T, 3, 1> spl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
    Eigen::Matrix<T, 3, 1> spm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
    // Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

    Eigen::Quaternion<T> q_ref{q1[0], q1[1], q1[2], q1[3]};
    Eigen::Matrix<T, 3, 1> t_ref{t1[0], t1[1], t1[2]};
    Eigen::Quaternion<T> q_src{q2[0], q2[1], q2[2], q2[3]};
    Eigen::Matrix<T, 3, 1> t_src{t2[0], t2[1], t2[2]};

    Eigen::Matrix<T, 3, 1> lp, lpj, lpl, lpm;
    lp = q_ref.inverse() * cp - q_ref.inverse() * t_ref;
    lpj = q_src.inverse() * spj - q_src.inverse() * t_src;
    lpl = q_src.inverse() * spl - q_src.inverse() * t_src;
    lpm = q_src.inverse() * spm - q_src.inverse() * t_src;

    Eigen::Matrix<T, 3, 1>  ljm_norm;
    ljm_norm = (lpj - lpl).cross(lpj - lpm);
    ljm_norm.normalize();
    
    T residual = (lp - lpj).dot(ljm_norm);

    return std::sqrt(residual*residual);
}


} // namespace sensemap

