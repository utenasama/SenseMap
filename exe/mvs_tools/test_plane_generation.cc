//Copyright (c) 2020, SenseTime Group.
//All rights reserved.

#include <iostream>
#include <iterator>
#include <list>
#include <queue>
#include <dirent.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <time.h>
#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/SVD>

#include "util/misc.h"
#include "util/ply.h"
#include "util/obj.h"
#include "util/endian.h"
#include "util/types.h"
#include "util/threading.h"
#include "util/mat.h"
#include "util/semantic_table.h"
#include "util/mesh_info.h"
#include "base/common.h"
#include "mvs/workspace.h"
#include "mvs/depth_map.h"
#include "estimators/plane_estimator.h"
#include "optim/ransac/loransac.h"

#include "base/version.h"
#include "../Configurator_yaml.h"

using namespace sensemap;
using namespace sensemap::mvs;

std::string configuration_file_path;

struct NestPolygon {
    int poly_id;
    std::vector<int> inner_poly_ids;
};

struct BoundingBox2D {
    Eigen::Vector2d lt;
    Eigen::Vector2d rb;
};

namespace unionset {
void Init(std::vector<int>& par) {
    std::iota(par.begin(), par.end(), 0);
}

int Find(std::vector<int>& par, int x) {
    if (x != par[x]) {
        par[x] = Find(par, par[x]);
    }
    return par[x];
}

void UnionSet(std::vector<int>& par, std::vector<int>& rank, int x, int y) {
    x = Find(par, x);
    y = Find(par, y);
    if (x == y) {
        return;
    }
    if (rank[x] > rank[y]) {
        par[y] = x;
        rank[x] += rank[y];
    } else {
        par[x] = y;
        rank[y] += rank[x];
    }
}

}

namespace ParamOption {
int min_consistent_facet = 1000;
double dist_point_to_line = 0.0;
double angle_diff_thres = 20.0;
double dist_ratio_point_to_plane = 10.0;
double ratio_singlevalue_xz = 1500.0;
double ratio_singlevalue_yz = 400.0;
};

void EstimateMultiPlanes(const std::vector<Eigen::Vector3d>& points,
                         const std::vector<Eigen::Vector3d>& normals,
                         std::vector<int>& planes_idx,
                         std::vector<Eigen::Vector4d>& planes) {
    
    std::vector<Eigen::Vector3d> plane_points = points;
    std::vector<int> points_idx(points.size());
    std::iota(points_idx.begin(), points_idx.end(), 0);
    planes_idx.resize(points.size(), -1);

    RANSACOptions ransac_options;
    ransac_options.max_error = 0.01;
    LORANSAC<PlaneEstimator, PlaneLocalEstimator> ransac(ransac_options);

    planes.clear();
    while(!plane_points.empty()) {
        auto report = ransac.Estimate(plane_points);

        double mean_residual = report.support.residual_sum / report.support.num_inliers;
        double max_residual = std::sqrt(mean_residual) * 4;

        RANSACOptions local_ransac_options;
        local_ransac_options.max_error = max_residual;
        LORANSAC<PlaneLocalEstimator, PlaneLocalEstimator> local_ransac(local_ransac_options);
        auto local_report = local_ransac.Estimate(plane_points);

        size_t num_inlier = 0;
        std::for_each(local_report.inlier_mask.begin(), 
                      local_report.inlier_mask.end(), [&](char& mask) {
            num_inlier += (int)!!mask;
        });      

        if (num_inlier < 10000) {
            break;
        }

        size_t plane_idx = planes.size();
        size_t i, j;
        Eigen::Vector3d mean_normal(0, 0, 0);
        for (i = 0, j = 0; i < local_report.inlier_mask.size(); ++i) {
            auto mask = local_report.inlier_mask.at(i);
            if (!mask) {
                plane_points.at(j) = plane_points.at(i);
                points_idx.at(j) = points_idx.at(i);
                j = j + 1;
            } else {
                planes_idx.at(points_idx.at(i)) = plane_idx;
                mean_normal += normals.at(points_idx.at(i));
            }
        }
        plane_points.resize(j);
        points_idx.resize(j);

        mean_normal = (mean_normal / num_inlier).normalized();

        auto model = report.model;
        if (model.head<3>().dot(mean_normal) < 0) {
            model = -model;
        }
        planes.emplace_back(model);
        std::cout << "plane param: " << model.transpose() << std::endl;
        std::cout << "inlier vertices: " << num_inlier << std::endl;
    }
}

void EstimateConsistentMultiPlanes(const std::vector<Eigen::Vector3d>& points,
                                   const std::vector<Eigen::Vector3i>& facets,
                                   std::vector<int>& planes_idx,
                                   std::vector<Eigen::Vector4d>& planes) {

    const double radian_diff_thres = std::cos(ParamOption::angle_diff_thres / 180 * M_PI);
    size_t i, j;

    std::vector<std::vector<int> > adj_facets_per_vertex(points.size());
    std::vector<std::vector<int> > adj_facets_per_facet(facets.size());
    for (i = 0; i < facets.size(); ++i) {
        auto facet = facets.at(i);
        adj_facets_per_vertex.at(facet[0]).push_back(i);
        adj_facets_per_vertex.at(facet[1]).push_back(i);
        adj_facets_per_vertex.at(facet[2]).push_back(i);
    }
    for (i = 0; i < facets.size(); ++i) {
        auto facet = facets.at(i);
        std::unordered_set<int> adj_facets;
        for (j = 0; j < 3; ++j) {
            for (auto facet_id : adj_facets_per_vertex.at(facet[j])) {
                if (facet_id != i) {
                    adj_facets.insert(facet_id);
                }
            }
        }
        if (adj_facets.size() == 0) {
            continue;
        }
        std::copy(adj_facets.begin(), adj_facets.end(), std::back_inserter(adj_facets_per_facet.at(i)));
    }

    // Compute face normals.
    std::vector<Eigen::Vector3d> face_normals(facets.size());
    for (i = 0; i < facets.size(); ++i) {
        auto facet = facets.at(i);
        auto& vtx0 = points.at(facet[0]);
        auto& vtx1 = points.at(facet[1]);
        auto& vtx2 = points.at(facet[2]);
        face_normals.at(i) = (vtx1 - vtx0).cross(vtx2 - vtx0).normalized();
    }

    std::vector<std::vector<int> > consistent_facets;
    std::vector<char> assigned(facets.size(), 0);
    planes_idx.resize(points.size(), -1);

    planes.clear();

    for (i = 0; i < facets.size(); ++i) {
        if (assigned.at(i)) {
            continue;
        }

        std::queue<int> Q;
        Q.push(i);
        assigned.at(i) = 1;

        auto m_normal = face_normals.at(i);
        auto f = facets.at(i);
        Eigen::Vector3d m_C = (points.at(f[0]) + points.at(f[1]) + points.at(f[2])) / 3;
        double m_dist = 0.0;

        std::vector<int> consistent_facets_;
        consistent_facets_.push_back(i);

        while(!Q.empty()) {
            auto facet_id = Q.front();
            Q.pop();

            for (auto adj_facet : adj_facets_per_facet.at(facet_id)) {
                if (assigned.at(adj_facet)) {
                    continue;
                }
                auto m_nNormal = (m_normal / consistent_facets_.size()).normalized();
                double angle = m_nNormal.dot(face_normals.at(adj_facet));
                if (angle < radian_diff_thres) {
                    continue;
                }
                auto f = facets.at(adj_facet);
                Eigen::Vector3d C = (points.at(f[0]) + points.at(f[1]) + points.at(f[2])) / 3;
                Eigen::Vector3d mm_C = m_C / consistent_facets_.size();
                double mm_dist = m_dist / consistent_facets_.size();
                double dist = std::fabs(m_nNormal.dot(C - mm_C));
                if (m_dist != 0 && 
                    dist > ParamOption::dist_ratio_point_to_plane * mm_dist) {
                    continue;
                }
                m_dist += dist;

                Q.push(adj_facet);

                assigned.at(adj_facet) = 1;
                consistent_facets_.push_back(adj_facet);
                m_normal += face_normals.at(adj_facet);
                m_C += C;
            }
        }
        
        if (consistent_facets_.size() == 0) {
            continue;
        }

        m_dist /= consistent_facets_.size();
        m_dist = m_dist == 0 ? 1e-3 : m_dist;

        std::vector<Eigen::Vector3d> facet_centaries;
        facet_centaries.reserve(consistent_facets_.size());
        for (auto facet_id : consistent_facets_) {
            auto f = facets.at(facet_id);
            auto C = (points.at(f[0]) + points.at(f[1]) + points.at(f[2])) / 3;
            facet_centaries.push_back(C);
        }

        RANSACOptions options;
        options.max_error = m_dist * ParamOption::dist_ratio_point_to_plane;
        LORANSAC<PlaneEstimator, PlaneLocalEstimator> estimator(options);
        auto report = estimator.Estimate(facet_centaries);
        if (!report.success) {
            continue;
        }
        auto model = report.model;
  
        j = 0;
        Eigen::Vector3d mean_C(0, 0, 0);
        for (int k = 0; k < consistent_facets_.size(); ++k) {
            if (report.inlier_mask.at(k)) {
                assigned.at(consistent_facets_[k]) = 1;
                consistent_facets_.at(j) = consistent_facets_.at(k);
                facet_centaries.at(j) = facet_centaries.at(k);
                mean_C += facet_centaries.at(j);
                j = j + 1;
            } else {
                assigned.at(consistent_facets_[k]) = 0;
            }
        }
        consistent_facets_.resize(j);
        facet_centaries.resize(j);
        if (consistent_facets_.size() < ParamOption::min_consistent_facet) {
            continue;
        }
        mean_C /= consistent_facets_.size();

#if 0
        double mean_dist = 0.0, stdev = 0.0;
        std::vector<double> dists;
        for (auto & C : facet_centaries) {
            double d = std::fabs(model.dot(C.homogeneous()));
            dists.push_back(d);
            mean_dist += d;
        }
        mean_dist /= dists.size();
        for (auto & dist : dists) {
            stdev += (dist - mean_dist) * (dist - mean_dist);
        }
        stdev = std::sqrt(stdev / dists.size());
        if (mean_dist / stdev > 1.26) {
            continue;
        }

        std::cout << planes.size() << ": " << mean_dist << " " << stdev << " " << mean_dist / stdev << std::endl;
#else
        Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
        for (j = 0; j < 3; ++j) {
            for (i = 0; i < facet_centaries.size(); ++i) {
                auto C = facet_centaries.at(i) - mean_C;
                M(j, 0) += C[j] * C.x();
                M(j, 1) += C[j] * C.y();
                M(j, 2) += C[j] * C.z();
            }
        }
        M /= facet_centaries.size();

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
        auto single_values = svd.singularValues().transpose();
        // std::cout << "single_values: " << single_values << std::endl;
        double xz_ratio = single_values.x() / single_values.z();
        double yz_ratio = single_values.y() / single_values.z();
        double xy_ratio = single_values.x() / single_values.y();
        // std::cout << "  => " << planes.size() << " " << xz_ratio << " " << yz_ratio << " " << xy_ratio << std::endl;

        if (xz_ratio < ParamOption::ratio_singlevalue_xz || 
            yz_ratio < ParamOption::ratio_singlevalue_yz) {
            continue;
        }
#endif
        consistent_facets.emplace_back(consistent_facets_);

        m_normal.normalize();
        if (model.head<3>().dot(m_normal) < 0) {
            model = -model;
        }
        planes.emplace_back(model);

    }

    for (i = 0; i < consistent_facets.size(); ++i) {
        std::cout << "consistent_facets.size(): " << consistent_facets.at(i).size() << std::endl;

        std::unordered_set<int> vtx_index_map;
        std::vector<Eigen::Vector3d> consistent_points;
        for (auto facet_id : consistent_facets.at(i)) {
            auto facet = facets.at(facet_id);
            if (vtx_index_map.find(facet[0]) == vtx_index_map.end()) {
                vtx_index_map.insert(facet[0]);
                planes_idx.at(facet[0]) = i;
                consistent_points.emplace_back(points.at(facet[0]));
            }
            if (vtx_index_map.find(facet[1]) == vtx_index_map.end()) {
                vtx_index_map.insert(facet[1]);
                planes_idx.at(facet[1]) = i;
                consistent_points.emplace_back(points.at(facet[1]));
            }
            if (vtx_index_map.find(facet[2]) == vtx_index_map.end()) {
                vtx_index_map.insert(facet[2]);
                planes_idx.at(facet[2]) = i;
                consistent_points.emplace_back(points.at(facet[2]));
            }
        }
    }
}

void GeneratePlaneRegions(const TriangleMesh mesh, 
                          std::vector<TriangleMesh>& plane_meshs,
                          std::vector<Eigen::Vector4d>& planes,
                          const bool export_ground = false) {
    int i, j, vtx_index = 0;
    std::unordered_map<int, int> vtx_index_map;
    std::vector<Eigen::Vector3i> plane_facets;
    plane_facets.reserve(mesh.faces_.size());
    std::vector<Eigen::Vector3d> plane_vertexs;
    plane_vertexs.reserve(mesh.vertices_.size());
    std::vector<Eigen::Vector3d> vertex_normals;
    vertex_normals.reserve(mesh.vertices_.size());

    bool has_label = mesh.vertex_labels_.size() != 0;

    int num_facet_ground = 0;
    for (i = 0; i < mesh.faces_.size(); ++i) {
        auto facet = mesh.faces_.at(i);

        if (has_label && export_ground) {
            auto label0 = mesh.vertex_labels_.at(facet[0]);
            auto label1 = mesh.vertex_labels_.at(facet[1]);
            auto label2 = mesh.vertex_labels_.at(facet[2]);
            if (label0 != LABLE_GROUND || label1 != LABLE_GROUND || 
                label2 != LABLE_GROUND) {
                continue;
            }
        }
        
        plane_facets.emplace_back(facet);

        auto& vtx0 = mesh.vertices_.at(facet[0]);
        auto& vtx_normal0 = mesh.vertex_normals_.at(facet[0]);
        auto& vtx1 = mesh.vertices_.at(facet[1]);
        auto& vtx_normal1 = mesh.vertex_normals_.at(facet[1]);
        auto& vtx2 = mesh.vertices_.at(facet[2]);
        auto& vtx_normal2 = mesh.vertex_normals_.at(facet[2]);
        if (vtx_index_map.find(facet[0]) == vtx_index_map.end()) {
            plane_vertexs.emplace_back(vtx0);
            vertex_normals.emplace_back(vtx_normal0);
            vtx_index_map[facet[0]] = vtx_index++;
        }
        if (vtx_index_map.find(facet[1]) == vtx_index_map.end()) {
            plane_vertexs.emplace_back(vtx1);
            vertex_normals.emplace_back(vtx_normal1);
            vtx_index_map[facet[1]] = vtx_index++;
        }
        if (vtx_index_map.find(facet[2]) == vtx_index_map.end()) {
            plane_vertexs.emplace_back(vtx2);
            vertex_normals.emplace_back(vtx_normal2);
            vtx_index_map[facet[2]] = vtx_index++;
        }
        
        num_facet_ground++;
    }

    if (export_ground && num_facet_ground == 0) {
        std::cout << "No ground facets!" << std::endl;
        return;
    }

    std::cout << StringPrintf("%d Vertices\n", plane_vertexs.size());

    for (auto& facet : plane_facets) {
        facet[0] = vtx_index_map.at(facet[0]);
        facet[1] = vtx_index_map.at(facet[1]);
        facet[2] = vtx_index_map.at(facet[2]);
    }

    std::vector<int> planes_idx;

    // EstimateMultiPlanes(plane_vertexs, vertex_normals, planes_idx, planes);
    EstimateConsistentMultiPlanes(plane_vertexs, plane_facets, planes_idx, planes);

    std::cout << "planes.size(): " << planes.size() << std::endl;

    plane_meshs.clear();
    plane_meshs.resize(planes.size());
    for (i = 0, j = 0; i < plane_facets.size(); ++i) {
        auto facet = plane_facets.at(i);
        auto plane_idx0 = planes_idx.at(facet[0]);
        auto plane_idx1 = planes_idx.at(facet[1]);
        auto plane_idx2 = planes_idx.at(facet[2]);
        if (plane_idx0 == -1 || 
            plane_idx0 != plane_idx1 || plane_idx0 != plane_idx2) {
            continue;
        }

        auto vtx0 = plane_vertexs.at(facet[0]);
        auto vtx1 = plane_vertexs.at(facet[1]);
        auto vtx2 = plane_vertexs.at(facet[2]);
        auto facet_normal = (vtx1 - vtx0).cross(vtx2 - vtx0).normalized();
        if (facet_normal.dot(planes.at(plane_idx0).head<3>()) < 0.7) {
            continue;
        }

        plane_meshs.at(plane_idx0).faces_.emplace_back(facet);
    }

    for (i = 0, j = 0; i < plane_meshs.size(); ++i) {
        TriangleMesh& mesh = plane_meshs.at(i);

        vtx_index = 0;
        vtx_index_map.clear();

        for (auto& facet : mesh.faces_) {
            auto vtx0 = plane_vertexs.at(facet[0]);
            auto vtx1 = plane_vertexs.at(facet[1]);
            auto vtx2 = plane_vertexs.at(facet[2]);
            
            if (vtx_index_map.find(facet[0]) == vtx_index_map.end()) {
                mesh.vertices_.emplace_back(vtx0);
                vtx_index_map[facet[0]] = vtx_index++;
            }
            if (vtx_index_map.find(facet[1]) == vtx_index_map.end()) {
                mesh.vertices_.emplace_back(vtx1);
                vtx_index_map[facet[1]] = vtx_index++;
            }
            if (vtx_index_map.find(facet[2]) == vtx_index_map.end()) {
                mesh.vertices_.emplace_back(vtx2);
                vtx_index_map[facet[2]] = vtx_index++;
            }
        }
        for (auto& facet : mesh.faces_) {
            facet[0] = vtx_index_map.at(facet[0]);
            facet[1] = vtx_index_map.at(facet[1]);
            facet[2] = vtx_index_map.at(facet[2]);
        }
    }
    std::cout << "Consistent plane meshes: " << plane_meshs.size() << std::endl;
}

void GenerateBorderList(TriangleMesh& mesh,
                        std::vector<std::vector<int> >& border_vertices) {
    MeshInfo mesh_info;
    mesh_info.initialize(mesh);

    size_t num_vert = mesh.vertices_.size();
    for (size_t i = 0; i < num_vert; ++i) {
        auto& vert = mesh.vertices_.at(i);
        MeshInfo::VertexInfo& vert_info = mesh_info.at(i);
        if (vert_info.vclass != MeshInfo::VERTEX_CLASS_COMPLEX) {
            continue;
        }
        std::vector<std::vector<int> > adj_facets;
        mesh_info.get_clustered_adj_faces(mesh, i, adj_facets);
        for (size_t j = 1; j < adj_facets.size(); ++j) {
            int vert_id = mesh.vertices_.size();
            mesh.vertices_.emplace_back(vert);

            for (auto f_id : adj_facets.at(j)) {
                auto& facet = mesh.faces_.at(f_id);
                if (facet.x() == i) {
                    facet.x() = vert_id;
                }
                if (facet.y() == i) {
                    facet.y() = vert_id;
                }
                if (facet.z() == i) {
                    facet.z() = vert_id;
                }
            }
        }
    }

    mesh_info.initialize(mesh);

    border_vertices.clear();
    mesh_info.get_border_lists(border_vertices);
}

Eigen::Matrix3d TransformToPlaneAxis(const Eigen::Vector4d& plane) {
    
    Eigen::Vector3d tmpv;
    if (plane.x() != 0 || plane.y() != 0) {
        tmpv[0] = plane.y();
        tmpv[1] = -plane.x();
        tmpv[2] = 0;
    } else {
        tmpv[0] = 0;
        tmpv[1] = plane.z();
        tmpv[2] = -plane.y();
    }
    Eigen::Vector3d zaxis(plane.data());
    Eigen::Vector3d yaxis = zaxis.cross(tmpv).normalized();
    Eigen::Vector3d xaxis = yaxis.cross(zaxis);
    Eigen::Matrix3d transformation;
    // transformation.row(0) = xaxis;
    // transformation.row(1) = yaxis;
    // transformation.row(2) = zaxis;
    // Swap x and y axis.
    transformation.row(0) = yaxis;
    transformation.row(1) = xaxis;
    transformation.row(2) = -zaxis;

    return transformation;
}

bool PointInRegion(const Eigen::Vector3d& pt, const std::vector<Eigen::Vector3d>& plist) {
	int nCross = 0;
	for (int i = 0; i < plist.size(); i++) {
		Eigen::Vector3d p1;
		Eigen::Vector3d p2;
		p1 = plist[i];
		p2 = plist[(i+1)%plist.size()];

        double x1 = p1.x();
        double y1 = p1.y();
        double x2 = p2.x();
        double y2 = p2.y();

		if (y1 == y2)
			continue;   //如果这条边是水平的，跳过
		if ( pt.y() < std::min(y1, y2)) //如果目标点低于这个线段，跳过
			continue; 
		if ( pt.y() >= std::max(y1, y2)) //如果目标点高于这个线段，跳过
			continue; 
		double x = (pt.y() - y1) * (x2 - x1) / (y2 - y1) + x1; 
		if ( x > pt.x() ) 
			nCross++; //如果交点在右边，统计加一。这等于从目标点向右发一条射线（ray），与多边形各边的相交（crossing）次数
	} 
    return nCross % 2;
}

bool PolygonInRegion(const std::vector<Eigen::Vector3d>& plist1,
                     const std::vector<Eigen::Vector3d>& plist2) {
    bool in_region = true;
    for (auto point : plist2) {
        if (!PointInRegion(point, plist1)) {
            in_region = false;
            break;
        }        
    }
    return in_region;
}

bool IntersectionBBox2D(const BoundingBox2D& bbox1, 
                        const BoundingBox2D& bbox2) {
    bool out =
        (bbox1.rb.x() < bbox2.lt.x()) || (bbox1.rb.y() < bbox2.lt.y()) ||
        (bbox1.lt.x() > bbox2.rb.x()) || (bbox1.lt.y() > bbox2.rb.y()) ||
        (bbox2.rb.x() < bbox1.lt.x()) || (bbox2.rb.y() < bbox1.lt.y()) ||
        (bbox2.lt.x() > bbox1.rb.x()) || (bbox2.lt.y() > bbox1.rb.y());
    return !out;
}

void GeneratePlaneTopology(
    std::vector<std::vector<Eigen::Vector3d> >& plane_points,
    std::vector<NestPolygon>& polys,
    const Eigen::Vector4d& plane) {

    Eigen::Vector3d plane_normal = plane.head<3>();
    Eigen::Matrix3d T = TransformToPlaneAxis(plane);

    size_t i, j;
    size_t num_poly = plane_points.size();

    std::vector<BoundingBox2D> bboxs2D(num_poly);
    for (i = 0; i < num_poly; ++i) {
        auto& lt = bboxs2D[i].lt;
        auto& rb = bboxs2D[i].rb;
        lt.x() = lt.y() = FLT_MAX;
        rb.x() = rb.y() = -FLT_MAX;
        for (auto & point : plane_points.at(i)) {
            double dist = plane.dot(point.homogeneous());
            point = T * (point - plane_normal * dist);

            lt.x() = std::min(lt.x(), point.x());
            lt.y() = std::min(lt.y(), point.y());
            rb.x() = std::max(rb.x(), point.x());
            rb.y() = std::max(rb.y(), point.y());
        }
    }

    std::vector<int> outer_id_map(num_poly, -1);
    for (i = 0; i < num_poly; ++i) {
        auto& bbox1 = bboxs2D.at(i);
        auto& points1 = plane_points.at(i);
        for (j = i + 1; j < num_poly; ++j) {
            auto& bbox2 = bboxs2D.at(j);
            auto& points2 = plane_points.at(j);

            if (!IntersectionBBox2D(bbox1, bbox2)) {
                continue;
            }
            if (PolygonInRegion(points1, points2)) {
                int& outer_id = outer_id_map[j];
                if (outer_id == -1 ||
                    PolygonInRegion(plane_points.at(outer_id), points1)) {
                    outer_id = i;
                }
            } else if (PolygonInRegion(points2, points1)) {
                int& outer_id = outer_id_map[i];
                if (outer_id == -1 ||
                    PolygonInRegion(plane_points.at(outer_id), points2)) {
                    outer_id = j;
                }
            }
        }
    }

    std::unordered_map<int, int> merged_poly;
    polys.resize(num_poly);

    for (i = 0; i < num_poly; ++i) {
        int path = 0;
        j = i;
        while((j = outer_id_map.at(j)) != -1) {
            path = path + 1;
        }
        if (path % 2 == 0) {
            polys[i].poly_id = i;
        } else if (path % 2 == 1) {
            int outer_id = outer_id_map.at(i);
            polys[outer_id].poly_id = outer_id;
            polys[outer_id].inner_poly_ids.push_back(i);
            merged_poly[i] = outer_id;
        }
    }

    for (i = 0, j = 0; i < num_poly; ++i) {
        if (merged_poly.count(i) != 0) {
            continue;
        }
        polys.at(j) = polys.at(i);
        j = j + 1;
    }
    polys.resize(j);
}

double PerpendicularDistance(const Eigen::Vector3d& p, 
                             const Eigen::Vector3d& p1, 
                             const Eigen::Vector3d& p2) {
    Eigen::Vector3d vec = (p2 - p1).normalized();
    double plen = (p - p1).dot(vec);
    Eigen::Vector3d pvec = vec * plen;
    return ((p - p1) - pvec).norm();
}

void RamerDouglasPeucker(const TriangleMesh& plane_mesh, 
                        const std::vector<int>& border_vertices,
                        const double epsilon,
                        std::vector<int>& out_vertices,
                        Eigen::RowMatrix3x4d trans = Eigen::RowMatrix3x4d::Identity()) {
    if (border_vertices.size() < 2) {
        return;
    }

    auto & vertices = plane_mesh.vertices_;

    double dmax = 0.0;
    size_t index = 0;
    size_t end = border_vertices.size() - 1;
    for (size_t i = 1; i < end; ++i) {
        auto pi = trans * vertices.at(border_vertices[i]).homogeneous();
        auto p0 = trans * vertices.at(border_vertices[0]).homogeneous();
        auto pe = trans * vertices.at(border_vertices[end]).homogeneous();
        double d = PerpendicularDistance(pi, p0, pe);
		if (d > dmax) {
			index = i;
			dmax = d;
		}
    }
    if (dmax > epsilon) {
        // Recursive call
		std::vector<int> outs1;
		std::vector<int> outs2;
		std::vector<int> firstLine(border_vertices.begin(), border_vertices.begin() + index + 1);
		std::vector<int> lastLine(border_vertices.begin() + index, border_vertices.end());
		RamerDouglasPeucker(plane_mesh, firstLine, epsilon, outs1);
		RamerDouglasPeucker(plane_mesh, lastLine, epsilon, outs2);

		// Build the result list
		out_vertices.assign(outs1.begin(), outs1.end() - 1);
		out_vertices.insert(out_vertices.end(), outs2.begin(), outs2.end());
    } else {
        //Just return start and end points
		out_vertices.clear();
		out_vertices.push_back(border_vertices[0]);
		out_vertices.push_back(border_vertices[end]);
    }
}

void WritePolys(FILE* file, 
                int& g_poly_id,
                const int plane_idx,
                const Eigen::Vector4d& plane,
                const TriangleMesh& plane_mesh,
                const std::vector<NestPolygon>& polys,
                const std::vector<std::vector<int> >& border_vertices,
                const bool verbose = false) {
    // plane parameter.
    fprintf(file, "# plane parameter.\n");
    fprintf(file, "%f %f %f %f\n", plane.x(), plane.y(), plane.z(), plane.w());

    fprintf(file, "\n# count of outer polygon.\n");
    fprintf(file, "%d\n", polys.size());

    for (int j = 0; j < polys.size(); ++j) {
        int border_id = polys.at(j).poly_id;
        auto& border_vtx_ids = border_vertices.at(border_id);

        auto inner_border_ids = polys.at(j).inner_poly_ids;

        // polygon id.
        fprintf(file, "\n# polygon id.\n");
        fprintf(file, "ID: %d\n", g_poly_id++);
        
        fprintf(file, "# polygon vertex list(x y z list).\n");
        fprintf(file, "%d\n", border_vtx_ids.size());

        double cx = std::rand() * 1.0 / RAND_MAX * 255;
        double cy = std::rand() * 1.0 / RAND_MAX * 255;
        double cz = std::rand() * 1.0 / RAND_MAX * 255;

        for (auto vtx_id : border_vtx_ids) {
            auto vtx = plane_mesh.vertices_.at(vtx_id);

            // double dist = plane.dot(vtx.homogeneous());
            // vtx = vtx - plane.head<3>() * dist;

            fprintf(file, "%f %f %f\n", vtx.x(), vtx.y(), vtx.z());
        }
        fprintf(file, "\n");

        fprintf(file, "# count of inner polygon.\n");
        fprintf(file, "%d\n", inner_border_ids.size());

        for (auto inner_border_id : inner_border_ids) {
            auto& inner_border_vtx_ids = border_vertices.at(inner_border_id);

            fprintf(file, "\n# inner polygon id.\n");
            fprintf(file, "ID: %d\n", g_poly_id++);
            fprintf(file, "# polygon vertex list(x y z list).\n");
            fprintf(file, "%d\n", inner_border_vtx_ids.size());

            cx = std::rand() * 1.0 / RAND_MAX * 255;
            cy = std::rand() * 1.0 / RAND_MAX * 255;
            cz = std::rand() * 1.0 / RAND_MAX * 255;

            for (auto vtx_id : inner_border_vtx_ids) {
                auto vtx = plane_mesh.vertices_.at(vtx_id);

                // double dist = plane.dot(vtx.homogeneous());
                // vtx = vtx - plane.head<3>() * dist;

                fprintf(file, "%f %f %f\n", vtx.x(), vtx.y(), vtx.z());
            }
            fprintf(file, "\n");
        }
    }
}

void PlaneGenerator(const TriangleMesh &mesh, 
                    const std::string &workspace_path,
                    const bool export_ground = false,
                    const bool verbose = false) {
    std::vector<TriangleMesh> plane_meshs;
    std::vector<Eigen::Vector4d> planes;
    GeneratePlaneRegions(mesh, plane_meshs, planes, export_ground);

    std::srand((unsigned)time(NULL));

    FILE *poly_file = fopen(StringPrintf("%s/polys.txt", workspace_path.c_str()).c_str(), "w");
    fprintf(poly_file, "\n");
    fprintf(poly_file, "%d\n", plane_meshs.size());

    TriangleMesh border_mesh;
    int g_poly_id = 0, g_poly_id_trans = 0;
    for (size_t plane_idx = 0; plane_idx < plane_meshs.size(); ++plane_idx) {
        auto plane_mesh = plane_meshs.at(plane_idx);

        std::vector<std::vector<int> > border_vertices;
        GenerateBorderList(plane_mesh, border_vertices);

        std::sort(border_vertices.begin(), border_vertices.end(), 
            [&](const std::vector<int>& list1, 
                const std::vector<int>& list2) {
            return list1.size() > list2.size();
        });

        size_t i, j;

        for (i = 0, j = 0; i < border_vertices.size(); ++i) {
            if (border_vertices.at(i).size() < 50) {
                continue;
            }

            std::vector<int> out_vertices;
            if (export_ground) {
                RamerDouglasPeucker(plane_mesh, border_vertices.at(i), 
                                    2 * ParamOption::dist_point_to_line,
                                    out_vertices);
            } else {
                RamerDouglasPeucker(plane_mesh, border_vertices.at(i), 
                                    ParamOption::dist_point_to_line,
                                    out_vertices);
            }
            std::cout << "reduce vertices: " << border_vertices.at(i).size() << " => " << out_vertices.size() << std::endl;

            if (out_vertices.size() < 3) {
                continue;
            }

            border_vertices[j] = out_vertices;
            j = j + 1;
        }
        border_vertices.resize(j);

        std::vector<std::vector<Eigen::Vector3d> > plane_points;
        plane_points.resize(border_vertices.size());
        for (i = 0; i < border_vertices.size(); ++i) {
            std::vector<Eigen::Vector3d> plane_points_;
            for (j = 0; j < border_vertices.at(i).size(); ++j) {
                auto vtx_id = border_vertices.at(i)[j];
                auto vtx = plane_mesh.vertices_.at(vtx_id);
                plane_points_.emplace_back(vtx);
            }
            plane_points[i] = plane_points_;
        }

        Eigen::Vector4d plane = planes.at(plane_idx);

        std::vector<NestPolygon> polys;
        GeneratePlaneTopology(plane_points, polys, plane);

        std::cout << "polys.size(): " << polys.size() << std::endl;
        {
            WritePolys(poly_file, g_poly_id, plane_idx, plane, plane_mesh, polys, border_vertices, verbose);
            ;
        }
        for (j = 0; j < polys.size(); ++j) {
            int border_id = polys.at(j).poly_id;
            auto& border_vtx_ids = border_vertices.at(border_id);
            auto inner_border_ids = polys.at(j).inner_poly_ids;

            Eigen::Vector3d color;
            color.x() = std::rand() * 1.0 / RAND_MAX * 255;
            color.y() = std::rand() * 1.0 / RAND_MAX * 255;
            color.z() = std::rand() * 1.0 / RAND_MAX * 255;

            for (auto vtx_id : border_vtx_ids) {
                auto vtx = plane_mesh.vertices_.at(vtx_id);
                border_mesh.vertices_.emplace_back(vtx);
                border_mesh.vertex_colors_.emplace_back(color);
            }
            for (auto inner_border_id : inner_border_ids) {
                auto& inner_border_vtx_ids = border_vertices.at(inner_border_id);
                for (auto vtx_id : inner_border_vtx_ids) {
                    auto vtx = plane_mesh.vertices_.at(vtx_id);             border_mesh.vertices_.emplace_back(vtx);
                    border_mesh.vertex_colors_.emplace_back(color);
                }
            }
        }
    }
    fclose(poly_file);

    if (verbose) {

        WriteTriangleMeshObj(
            StringPrintf("%s/plane_borders.obj", workspace_path.c_str()), 
            border_mesh, true, false);
        
        TriangleMesh all_plane_meshs;
        for (auto& plane_mesh : plane_meshs) {
            
            // Colorizing plane mesh.
            double cx = std::rand() * 1.0 / RAND_MAX * 255;
            double cy = std::rand() * 1.0 / RAND_MAX * 255;
            double cz = std::rand() * 1.0 / RAND_MAX * 255;
            Eigen::Vector3d rgb(cx, cy, cz);

            std::unordered_map<int, int> vtx_index_map;
            for (auto facet : plane_mesh.faces_) {
                int idx0, idx1, idx2;
                if (vtx_index_map.find(facet[0]) == vtx_index_map.end()) {
                    idx0 = all_plane_meshs.vertices_.size();
                    auto vtx = plane_mesh.vertices_.at(facet[0]);
                    all_plane_meshs.vertices_.emplace_back(vtx);
                    all_plane_meshs.vertex_colors_.emplace_back(rgb);
                    vtx_index_map[facet[0]] = idx0;
                } else {
                    idx0 = vtx_index_map.at(facet[0]);
                }
                if (vtx_index_map.find(facet[1]) == vtx_index_map.end()) {
                    idx1 = all_plane_meshs.vertices_.size();
                    auto vtx = plane_mesh.vertices_.at(facet[1]);
                    all_plane_meshs.vertices_.emplace_back(vtx);
                    all_plane_meshs.vertex_colors_.emplace_back(rgb);
                    vtx_index_map[facet[1]] = idx1;
                } else {
                    idx1 = vtx_index_map.at(facet[1]);
                }
                if (vtx_index_map.find(facet[2]) == vtx_index_map.end()) {
                    idx2 = all_plane_meshs.vertices_.size();
                    auto vtx = plane_mesh.vertices_.at(facet[2]);
                    all_plane_meshs.vertices_.emplace_back(vtx);
                    all_plane_meshs.vertex_colors_.emplace_back(rgb);
                    vtx_index_map[facet[2]] = idx2;
                } else {
                    idx2 = vtx_index_map.at(facet[2]);
                }
                facet[0] = idx0;
                facet[1] = idx1;
                facet[2] = idx2;
                all_plane_meshs.faces_.emplace_back(facet);
            }
        }

        WriteTriangleMeshObj(
            StringPrintf("%s/plane_mesh.obj", workspace_path.c_str()), 
            all_plane_meshs, true, false);
    }
}

int main(int argc, char *argv[]) {

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

	std::string workspace_path = param.GetArgument("workspace_path", "");
	std::string image_type = param.GetArgument("image_type", "perspective");
	std::string trans_path = param.GetArgument("trans_path", "");
    bool export_ground = param.GetArgument("export_ground", 1);
    bool verbose = param.GetArgument("verbose", 0);
    
    ParamOption::min_consistent_facet = param.GetArgument("min_consistent_facet", 1000);
    ParamOption::dist_point_to_line = param.GetArgument("dist_point_to_line", 0.f);
    ParamOption::angle_diff_thres = param.GetArgument("angle_diff", 20.0f);
    ParamOption::dist_ratio_point_to_plane = param.GetArgument("dist_ratio_point_to_plane", 10.0f);
    ParamOption::ratio_singlevalue_xz = param.GetArgument("ratio_singlevalue_xz", 1500.0f);
    ParamOption::ratio_singlevalue_yz = param.GetArgument("ratio_singlevalue_yz", 400.0f);

    std::string semantic_table_path = param.GetArgument("semantic_table_path", "");
    if (ExistsFile(semantic_table_path)) {
        LoadSemanticColorTable(semantic_table_path.c_str());
    }

    int num_reconstruction = 0;
    for (size_t reconstruction_idx = 0; ; reconstruction_idx++) {
        auto reconstruction_path =
            JoinPaths(workspace_path, std::to_string(reconstruction_idx));
        if (!ExistsDir(reconstruction_path)) {
            break;
        }

        auto dense_reconstruction_path =
            JoinPaths(reconstruction_path, DENSE_DIR);
        if (!ExistsDir(dense_reconstruction_path)) {
            break;
        }

        auto model_name = std::string(MODEL_NAME);
        model_name = model_name.substr(0, model_name.size() - 4);
        auto in_model_path = JoinPaths(dense_reconstruction_path, model_name + ".obj");
        auto in_model_sem_path = JoinPaths(dense_reconstruction_path, model_name + "_sem.obj");

        bool has_sem = false;
        TriangleMesh mesh;
        if (ExistsFile(in_model_sem_path)) {
            std::cout << in_model_sem_path << std::endl;
            ReadTriangleMeshObj(in_model_sem_path, mesh, true, true);
            has_sem = true;
        } else if (ExistsFile(in_model_path)) {
            std::cout << in_model_path << std::endl;
            ReadTriangleMeshObj(in_model_path, mesh, true, false);
            has_sem = false;
        } else {
            continue;
        }

        Eigen::RowMatrix3x4d trans;
        bool has_trans = ExistsFile(trans_path);
        std::cout << trans_path << std::endl;
        if (has_trans) {
            // Load transform matrix
            cv::FileStorage fs;
            fs.open(trans_path, cv::FileStorage::READ);
            cv::Mat trans_mat;
            if(fs["transMatrix"].type() != cv::FileNode::MAP){
                std::cout << "ERROR: Input yaml error !!" << std::endl;
                exit(-1);
            }
            fs["transMatrix"] >> trans_mat;

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 4; j++) {
                    trans(i, j) = trans_mat.at<double>(i, j);
                }
            }   

            std::cout << "trans: " << std::endl << trans << std::endl;

            for (size_t i = 0; i < mesh.vertices_.size(); ++i) {
                Eigen::Vector4d vtx = mesh.vertices_[i].homogeneous();
                mesh.vertices_[i] = trans * vtx;
                if (!mesh.vertex_normals_.empty()) {
                    Eigen::Vector3d nvtx = mesh.vertex_normals_[i];
                    mesh.vertex_normals_[i] = trans.block<3, 3>(0, 0) * nvtx;
                }
            }
        }

        PrintHeading2("Generating Multi Planes");
        std::string workspace_path = JoinPaths(dense_reconstruction_path, "multi-planes");
        CreateDirIfNotExists(workspace_path);
        PlaneGenerator(mesh, workspace_path, false, verbose);
        if (has_sem) {
            PrintHeading2("Generating Horizontal Planes");
            workspace_path = JoinPaths(dense_reconstruction_path, "horizontal_planes");
            CreateDirIfNotExists(workspace_path);
            PlaneGenerator(mesh, workspace_path, export_ground, verbose);
        }

        num_reconstruction++;
    }
    return 0;
}