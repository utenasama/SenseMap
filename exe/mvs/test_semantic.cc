//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <iostream>
#include <iterator>
#include <functional>
#include <list>
#include <dirent.h>
#include <sys/stat.h>
#include <boost/filesystem/path.hpp>
#include <malloc.h>
#include <jsoncpp/json/json.h>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_triangle_primitive.h>

#include "util/proc.h"
#include "util/misc.h"
#include "util/ply.h"
#include "util/obj.h"
#include "util/endian.h"
#include "util/types.h"
#include "util/threading.h"
#include "util/mat.h"
#include "util/mesh_info.h"
#include "util/semantic_table.h"
#include "util/timer.h"
#include "util/exception_handler.h"
#include "base/common.h"
#include "mvs/workspace.h"
#include "mvs/depth_map.h"

#include "base/version.h"
#include "../Configurator_yaml.h"

using namespace sensemap;
using namespace sensemap::mvs;

typedef CGAL::Simple_cartesian<double>                       Kernel;
typedef CGAL::Polyhedron_3<Kernel>                           Polyhedron;
typedef Polyhedron::HalfedgeDS                               HalfedgeDS;
typedef Kernel::Point_3                                      Point;
typedef Kernel::Vector_3                                     Vector;
typedef Kernel::Segment_3                                    Segment;
typedef Kernel::Ray_3                                        Ray;
typedef Kernel::Triangle_3                                   Triangle;
// typedef CGAL::AABB_face_graph_triangle_primitive<Polyhedron> Primitive;
typedef std::list<Triangle>::iterator                        Iterator;
typedef CGAL::AABB_triangle_primitive<Kernel, Iterator>      Primitive;
typedef CGAL::AABB_traits<Kernel, Primitive>                 Traits;
typedef CGAL::AABB_tree<Traits>                              Tree;
// typedef boost::optional< Tree::Intersection_and_primitive_id<Segment>::Type > Segment_intersection;
typedef Tree::Primitive_id                                   Primitive_id;

// A modifier creating a triangle with the incremental builder.
template <class HDS>
class Build_triangle : public CGAL::Modifier_base<HDS> {
public:
    Build_triangle(const TriangleMesh& mesh) : mesh_(mesh) {}
    void operator()( HDS& hds) {
        // Postcondition: hds is a valid polyhedral surface.
        CGAL::Polyhedron_incremental_builder_3<HDS> B( hds, true);

        typedef typename HDS::Vertex   Vertex;
        typedef typename Vertex::Point VPoint;

        B.begin_surface(mesh_.vertices_.size(), mesh_.faces_.size());
        for (auto vert : mesh_.vertices_) {
            B.add_vertex(VPoint(vert[0], vert[1], vert[2]));
        }
        for (auto facet : mesh_.faces_) {
            B.begin_facet();
            B.add_vertex_to_facet(facet[0]);
            B.add_vertex_to_facet(facet[1]);
            B.add_vertex_to_facet(facet[2]);
            B.end_facet();
        }
        B.end_surface();
    }
private:
    TriangleMesh mesh_;
};

namespace ParamOption {
int min_consistent_facet = 1000;
double dist_point_to_line = 0.0;
double angle_diff_thres = 20.0;
double dist_ratio_point_to_plane = 10.0;
double ratio_singlevalue_xz = 1500.0;
double ratio_singlevalue_yz = 400.0;
};

void LoadSemanticLabels(const std::string filepath, std::vector<uint8_t>& label_ids) {
    std::ifstream file;
    file.open(filepath.c_str(), std::ofstream::in);
    if (!file.is_open()) {
        std::cout << "Warning! Open Semantic Outlier File Failed!" << std::endl;
        return;
    }

    label_ids.clear();

    std::string line;
    std::string item;
    while (std::getline(file, line)) {
        StringTrim(&line);
        if (line.empty()) {
            continue;
        }
        std::stringstream line_stream(line);
        while (!line_stream.eof()) {
            std::getline(line_stream, item, ' ');
            label_ids.push_back(std::stoi(item));
        }
    }
    file.close();
}

void ModelSemantization(const std::vector<mvs::Image>& images,
    const std::vector<std::unique_ptr<Bitmap> >& semantic_maps,
    const TriangleMesh& mesh, TriangleMesh& sem_mesh) {

    const int invalid_label = -1;
    std::vector<std::set<size_t, std::greater<size_t>> > verts_vis(mesh.vertices_.size());

    for (size_t image_idx = 0; image_idx < images.size(); ++image_idx) {
        const mvs::Image& image = images.at(image_idx);
        const int width = image.GetWidth();
        const int height = image.GetHeight();
        const double thresh_fov = std::cos(90.0);

        Eigen::RowMatrix3d K = Eigen::RowMatrix3f(image.GetK()).cast<double>();
        Eigen::RowMatrix3d R = Eigen::RowMatrix3f(image.GetR()).cast<double>();
        Eigen::Vector3d T = Eigen::Vector3f(image.GetT()).cast<double>();

        Eigen::Vector3d C = -R.transpose() * T;
        Eigen::Vector3d ray = R.row(2);
        Eigen::RowMatrix3d Kinv = K.inverse();

        std::vector<float> depth_map(height * width);
        std::vector<int> vis_pxl_map(height * width, -1);

        for (size_t i = 0; i < mesh.faces_.size(); ++i) {
            if (ray.dot(mesh.face_normals_[i]) > 0.4) {
                continue;
            }
            const Eigen::Vector3i& facet = mesh.faces_[i];
            const Eigen::Vector3d p0 = mesh.vertices_[facet[0]];
            const Eigen::Vector3d p1 = mesh.vertices_[facet[1]];
            const Eigen::Vector3d p2 = mesh.vertices_[facet[2]];
            const Eigen::Vector3d centroid = (p0 + p1 + p2) / 3;
            
            const double dev_angle = ray.dot((centroid - C).normalized());
            // Outside of FOV.
            if (dev_angle < thresh_fov) {
                continue;
            }

            Eigen::Vector3d uv[3];
            Eigen::Vector3d Xc[3];
            Xc[0] = R * p0 + T;
            uv[0] = K * Xc[0]; uv[0] /= uv[0][2];
            Xc[1] = R * p1 + T;
            uv[1] = K * Xc[1]; uv[1] /= uv[1][2];
            Xc[2] = R * p2 + T;
            uv[2] = K * Xc[2]; uv[2] /= uv[2][2];
            
            int u_min = std::min(uv[0][0], std::min(uv[1][0], uv[2][0]));
            int u_max = std::max(uv[0][0], std::max(uv[1][0], uv[2][0]));
            int v_min = std::min(uv[0][1], std::min(uv[1][1], uv[2][1]));
            int v_max = std::max(uv[0][1], std::max(uv[1][1], uv[2][1]));
            u_min = std::max(0, u_min);
            v_min = std::max(0, v_min);
            u_max = std::min(width - 1, u_max);
            v_max = std::min(height - 1, v_max);
            if (Xc[0][2] < 0 || Xc[1][2] < 0 || Xc[2][2] < 0 ||
                u_min > u_max || v_min > v_max) {
                continue;
            }

#if 0
            Eigen::Vector3d Xo = R * centroid + T;
            float dc = Xo[2];

            for (int v = v_min; v <= v_max; ++v) {
                for (int u = u_min; u <= u_max; ++u) {
                    float d = depth_map[v * width + u];
                    if (d == 0 || dc < d) {
                        vis_pxl_map[v * width + u] = i;
                        depth_map[v * width + u] = dc;
                    }
                }
            }
#else
            Eigen::Vector3d edge1(Xc[1] - Xc[0]);
            Eigen::Vector3d edge2(Xc[2] - Xc[0]);

            const Eigen::Vector3d normal = edge1.cross(edge2).normalized();
            const Eigen::Vector3d normalPlane = normal * (1.0 / normal.dot(Xc[0]));

            for (int v = v_min; v <= v_max; ++v) {
                for (int u = u_min; u <= u_max; ++u) {
                    int t1 = (u - uv[0][0]) * (v - uv[1][1]) - (u - uv[1][0]) * (v - uv[0][1]);
                    int t2 = (u - uv[1][0]) * (v - uv[2][1]) - (u - uv[2][0]) * (v - uv[1][1]);
                    int t3 = (u - uv[2][0]) * (v - uv[0][1]) - (u - uv[0][0]) * (v - uv[2][1]);
                    if ((t1 >= 0 && t2 >= 0 && t3 >= 0) ||
                        (t1 <= 0 && t2 <= 0 && t3 <= 0)) {
                        Eigen::Vector3d xc = Kinv * Eigen::Vector3d(u, v, 1.0);
                        const double z = 1.0 / normalPlane.dot(xc);
                        const float nd = depth_map[v * width + u];
                        if (nd == 0 || z < nd) {
                            vis_pxl_map[v * width + u] = i;
                            depth_map[v * width + u] = z;
                        }
                    }
                }
            }
#endif
        }

        // mvs::Mat<float> mat(width, height, 1);
        // memcpy(mat.GetPtr(), depth_map.data(), sizeof(float) * width * height);
        // mvs::DepthMap depthmap(mat, 0, 100);
        // depthmap.ToBitmap().Write(StringPrintf("%06d.jpg", image_idx));

        std::unordered_set<size_t> vis_vtx_map;
        for (size_t i = 0; i < vis_pxl_map.size(); ++i) {
            int fidx = vis_pxl_map.at(i);
            if (fidx == -1) {
                continue;
            }
            auto facet = mesh.faces_.at(fidx);
            vis_vtx_map.insert(facet.x());
            vis_vtx_map.insert(facet.y());
            vis_vtx_map.insert(facet.z());
        }
        std::unordered_set<size_t>::iterator it = vis_vtx_map.begin();
        for (; it != vis_vtx_map.end(); ++it) {
            size_t vid = *it;

            Eigen::Vector3d proj = K * (R * mesh.vertices_.at(vid) + T);
            int u = proj[0] / proj[2];
            int v = proj[1] / proj[2];
            if (u < 0 || u >= width || v < 0 || v >= height) {
                continue;
            }

            int fidx = vis_pxl_map.at(v * width + u);
            if (fidx == -1) {
                continue;
            }
            Eigen::Vector3i facet = mesh.faces_.at(fidx);
            if (facet[0] != vid && facet[1] != vid && facet[2] != vid) {
                continue;
            }

            BitmapColor<uint8_t> semantic;
            semantic_maps.at(image_idx)->GetPixel(u, v, &semantic);
            uint8_t sid = semantic.r;
            verts_vis[vid].insert(sid);
        }

        std::cout << StringPrintf("\rProcess Image# %06d", image_idx + 1);
    }
    std::cout << std::endl;

    sem_mesh.vertex_colors_.clear();
    sem_mesh.vertex_labels_.clear();
    for (size_t i = 0; i < verts_vis.size(); ++i) {
        auto viss = verts_vis.at(i);
        if (viss.size() == 0) {
            sem_mesh.vertex_labels_.push_back(invalid_label);
            sem_mesh.vertex_colors_.emplace_back(Eigen::Vector3d(0, 0, 0));
            continue;
        }
        // std::cout << i << ": ";
        // std::set<size_t, std::greater<size_t>>::iterator it;
        // for (it = viss.begin(); it != viss.end(); ++it) {
        //     std::cout << *it << " ";
        // }
        // std::cout << std::endl;
        int best_label = *viss.begin();
        sem_mesh.vertex_labels_.emplace_back(best_label);
        Eigen::Vector3d rgb;
        rgb[0] = adepallete[best_label * 3];
        rgb[1] = adepallete[best_label * 3 + 1];
        rgb[2] = adepallete[best_label * 3 + 2];
        sem_mesh.vertex_colors_.emplace_back(rgb);
    }
}

void RefineSemantization(TriangleMesh& mesh) {
    std::cout << "RefineSemantization" << std::endl;
    size_t i, j;
    const double radian_diff_thres = std::cos(ParamOption::angle_diff_thres / 180 * M_PI);

    std::vector<Eigen::Vector3d>& points = mesh.vertices_;
    std::vector<int8_t>& labels = mesh.vertex_labels_;
    std::vector<Eigen::Vector3i>& facets = mesh.faces_;
    std::vector<Eigen::Vector3d>& face_normals = mesh.face_normals_;

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

    if (face_normals.empty()) {
        face_normals.resize(facets.size());
        for (i = 0; i < facets.size(); ++i) {
            auto& facet = facets.at(i);
            auto& vtx0 = points.at(facet[0]);
            auto& vtx1 = points.at(facet[1]);
            auto& vtx2 = points.at(facet[2]);
            face_normals.at(i) = (vtx1 - vtx0).cross(vtx2 - vtx0).normalized();
        }
    }

    std::vector<char> assigned(facets.size(), 0);
    for (i = 0; i < facets.size(); ++i) {
        if (assigned.at(i)) {
            continue;
        }

        int samps_per_label[256];
        memset(samps_per_label, 0, sizeof(int) * 256);

        std::queue<int> Q;
        Q.push(i);
        assigned.at(i) = 1;

        auto m_normal = face_normals.at(i);
        auto f = facets.at(i);
        Eigen::Vector3d m_C = (points.at(f[0]) + points.at(f[1]) + points.at(f[2])) / 3;
        double m_dist = 0.0;

        std::vector<int> consistent_facets_;
        consistent_facets_.push_back(i);
        std::unordered_set<int> consistent_verts_;

        while(!Q.empty()) {
            auto facet_id = Q.front();
            Q.pop();

            auto facet = facets.at(facet_id);
            if (consistent_verts_.find(facet[0]) == consistent_verts_.end()) {
                samps_per_label[(labels[facet[0]] + 256) % 256]++;
                consistent_verts_.insert(facet[0]);
            }
            if (consistent_verts_.find(facet[1]) == consistent_verts_.end()) {
                samps_per_label[(labels[facet[1]] + 256) % 256]++;
                consistent_verts_.insert(facet[1]);
            }
            if (consistent_verts_.find(facet[2]) == consistent_verts_.end()) {
                samps_per_label[(labels[facet[2]] + 256) % 256]++;
                consistent_verts_.insert(facet[2]);
            }

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
        
        if (consistent_facets_.size() < ParamOption::min_consistent_facet) {
            continue;
        }
        
        uint8_t best_label = -1;
        int num_vert_best_label = 0;
        for (int k = 0; k < 256; ++k) {
            if (samps_per_label[k] > num_vert_best_label) {
                num_vert_best_label = samps_per_label[k];
                best_label = k;
            }
        }
        if (best_label != -1) {
            for (auto& facet_id : consistent_facets_) {
                auto facet = facets.at(facet_id);
                labels.at(facet[0]) = best_label;
                labels.at(facet[1]) = best_label;
                labels.at(facet[2]) = best_label;
            }
        }
    }
}


void BuildSuperPixels(const TriangleMesh& mesh, 
                    const std::vector<std::vector<uint32_t> >& vert_visibilities,
                    // std::vector<uint32_t>& vertex_to_super_idx,
                    std::unordered_map<uint32_t, std::vector<uint32_t> > & super_idx_to_vectices,
                    // std::vector<std::unordered_set<uint32_t> >& neighbors_per_super,
                    std::vector<std::vector<uint32_t> >& super_visibilities) {
    Timer timer;
    timer.Start();

    const int num_vert = mesh.vertices_.size();

    std::vector<std::vector<size_t> > adj_vertices_per_vertex(num_vert);
    for (const auto & facet : mesh.faces_) {
        const int i0 = facet(0);
        const int i1 = facet(1);
        const int i2 = facet(2);
        auto & adj_verts0 = adj_vertices_per_vertex.at(i0);
        auto & adj_verts1 = adj_vertices_per_vertex.at(i1);
        auto & adj_verts2 = adj_vertices_per_vertex.at(i2);
        bool find = false;
        find = std::find_if(adj_verts0.begin(), adj_verts0.end(), [&](const int id) { return id == i1; }) != adj_verts0.end();
        if (!find) {
            if (adj_verts0.empty()) adj_verts0.push_back(i1);
            else adj_verts0.insert(std::lower_bound(adj_verts0.begin(), adj_verts0.end(), i1), i1);
            if (adj_verts1.empty()) adj_verts1.push_back(i0);
            else adj_verts1.insert(std::lower_bound(adj_verts1.begin(), adj_verts1.end(), i0), i0);
        }
        find = std::find_if(adj_verts0.begin(), adj_verts0.end(), [&](const int id) { return id == i2; }) != adj_verts0.end();
        if (!find) {
            if (adj_verts0.empty()) adj_verts0.push_back(i2);
            else adj_verts0.insert(std::lower_bound(adj_verts0.begin(), adj_verts0.end(), i2), i2);
            if (adj_verts2.empty()) adj_verts2.push_back(i0);
            else adj_verts2.insert(std::lower_bound(adj_verts2.begin(), adj_verts2.end(), i0), i0);
        }
        find = std::find_if(adj_verts1.begin(), adj_verts1.end(), [&](const int id) { return id == i2; }) != adj_verts1.end();
        if (!find) {
            if (adj_verts1.empty()) adj_verts1.push_back(i2);
            else adj_verts1.insert(std::lower_bound(adj_verts1.begin(), adj_verts1.end(), i2), i2);
            if (adj_verts2.empty()) adj_verts2.push_back(i1);
            else adj_verts2.insert(std::lower_bound(adj_verts2.begin(), adj_verts2.end(), i1), i1);
        }
    }

    int max_super_id = 0;
    super_idx_to_vectices[max_super_id].clear();
    std::vector<uint32_t> vertex_to_super_idx(num_vert, 0);

    for (size_t cid = 0; cid < num_vert; ++cid) {
        if (vert_visibilities.at(cid).empty() || vertex_to_super_idx.at(cid) != 0) {
            continue;
        }
        vertex_to_super_idx.at(cid) = ++max_super_id;
        super_idx_to_vectices[max_super_id].push_back(cid);

        std::queue<uint32_t> Q;
        Q.push(cid);
        while(!Q.empty()) {
            uint32_t cid = Q.front();
            Q.pop();

            auto viss = vert_visibilities.at(cid);
            // std::ostringstream oss;
            // std::copy(viss.begin(), viss.end(), std::ostream_iterator<int>(oss, ""));
            // std::string vis_str = oss.str(); 

            const auto & adj_verts = adj_vertices_per_vertex.at(cid);
            for (auto nid : adj_verts) {
                auto n_viss = vert_visibilities.at(nid);
                if (n_viss.empty() || vertex_to_super_idx.at(nid) != 0 || viss.size() != n_viss.size()) {
                    continue;
                }

                // std::ostringstream oss;
                // std::copy(n_viss.begin(), n_viss.end(), std::ostream_iterator<int>(oss, ""));
                // std::string nvis_str = oss.str(); 
                // if (nvis_str.compare(vis_str) == 0) {
                bool equal = true;
                for (int k = 0; k < viss.size(); ++k) {
                    if (viss[k] != n_viss[k]) {
                        equal = false;
                        break;
                    }
                }
                if (equal) {
                    vertex_to_super_idx.at(nid) = max_super_id;
                    super_idx_to_vectices[max_super_id].push_back(nid);
                    Q.push(nid);
                }
            }
        }
    }
    std::cout << "Number of super pixel: " << max_super_id << std::endl;

    // neighbors_per_super.resize(max_super_id + 1);
    super_visibilities.resize(max_super_id + 1);

    for (int super_id = 0; super_id < max_super_id + 1; ++super_id) {
        auto & vertex_ids = super_idx_to_vectices[super_id];
        vertex_ids.shrink_to_fit();
        if (!vertex_ids.empty()) {
            super_visibilities[super_id] = vert_visibilities[vertex_ids[0]];
        }
    }
    std::cout << StringPrintf("Construct Super Pixels cost %fmin\n", timer.ElapsedMinutes());
}

void ModelSemantization(const std::vector<mvs::Image>& images,
    const std::vector<std::string>& image_names,
    const std::string& workspace_path,
    const std::string& semantic_maps_path,
    const TriangleMesh& mesh, TriangleMesh& sem_mesh,
    const float max_ram) {
    const float thresh_fov = std::cos(70.0);
    const double estep = 0.005;
    std::vector<std::vector<uint32_t> > verts_vis;
    verts_vis.resize(mesh.vertices_.size());

    sem_mesh.vertex_colors_.clear();
    sem_mesh.vertex_labels_.clear();

#if 0
    std::cout << "Contruct Polyhedron" << std::endl;
    // Contruct Polyhedron.
    Polyhedron P;
    Build_triangle<HalfedgeDS> triangle(mesh);
    P.delegate(triangle);

    std::cout << "Construct AABB Tree" << std::endl;
    // Contruct AABB tree.
    Tree tree(CGAL::faces(P).first, CGAL::faces(P).second, P);
#else
    std::list<Triangle> triangles;
    for (auto facet : mesh.faces_) {
        auto a = mesh.vertices_.at(facet[0]);
        auto b = mesh.vertices_.at(facet[1]);
        auto c = mesh.vertices_.at(facet[2]);
        if ((a - b).norm() < 1e-6 || (b - c).norm() < 1e-6 || (a - c).norm() < 1e-6) {
            continue;
        }
        Triangle tri(Point(a[0], a[1], a[2]), Point(b[0], b[1], b[2]), 
                     Point(c[0], c[1], c[2]));
        triangles.emplace_back(tri);
    }

    Timer build_timer;
    build_timer.Start();
    std::cout << "Construct AABB Tree" << std::endl;
    // Contruct AABB tree.
    Tree tree(triangles.begin(), triangles.end());

    #ifdef CGAL_LINKED_WITH_TBB
      tree.build<CGAL::Parallel_tag>();
    #endif
    build_timer.PrintSeconds();
#endif

    size_t num_vert = mesh.vertices_.size();
    uint64_t indices_print_step = num_vert / 10 + 1;
    auto ComputeVertexVisibility = [&](const int i, int8_t* vert_label) {
        const auto& vert = mesh.vertices_[i];
        const auto& vert_normal = mesh.vertex_normals_[i];

        float max_samples = 0;
        int best_label = -1;
        for (size_t cam_id = 0; cam_id < images.size(); ++cam_id) {
            const mvs::Image& image = images[cam_id];
            Eigen::Map<const Eigen::Vector3f> cam_ray(image.GetViewingDirection());
            Eigen::Map<const Eigen::Vector3f> C(image.GetC());
            Eigen::Vector3f point_ray = (vert.cast<float>() - C).normalized();
            float cam_angle = cam_ray.dot(point_ray);
            if (cam_angle < thresh_fov) {
                continue;
            }

            Eigen::Map<const Eigen::RowMatrix3f> K(image.GetK());
            Eigen::Map<const Eigen::RowMatrix3f> R(image.GetR());
            Eigen::Map<const Eigen::Vector3f> T(image.GetT());

            Eigen::Vector3f proj = K * (R * vert.cast<float>() + T);
            int u = proj[0] / proj[2];
            int v = proj[1] / proj[2];
            if (u < 0 || u >= image.GetWidth() || 
                v < 0 || v >= image.GetHeight()) {
                continue;
            }
            
            // Construct segment query.
            Eigen::Vector3d query_point = vert - point_ray.cast<double>() * estep;
            Point a(query_point[0], query_point[1], query_point[2]);
            Point b(C[0], C[1], C[2]);
            Segment segment_query(a, b);

            // Test intersection with segment query.
            if (tree.do_intersect(segment_query)) {
                continue;
            }

            verts_vis[i].push_back(cam_id);
        }
        if (i % indices_print_step == 0) {
            std::cout << StringPrintf("\rProcess Point %d/%d", i, num_vert);
        }
    };

    const int num_eff_threads = GetEffectiveNumThreads(-1);
    std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
    std::unique_ptr<ThreadPool> thread_pool;
    thread_pool.reset(new ThreadPool(num_eff_threads));

    sem_mesh.vertex_labels_.resize(num_vert, -1);
    for (size_t i = 0; i < num_vert; ++i) {
        thread_pool->AddTask(ComputeVertexVisibility, i, &sem_mesh.vertex_labels_[i]); 
    }
    thread_pool->Wait();
    std::cout << std::endl;

    float available_memeory;
    GetAvailableMemory(available_memeory);
    std::cout << "Available Memory: " << available_memeory << "GB" << std::endl;

    std::list<Triangle>().swap(triangles);
    for (auto & viss : verts_vis) {
        viss.shrink_to_fit();
    }
    verts_vis.shrink_to_fit();

    GetAvailableMemory(available_memeory);
    std::cout << "Available Memory[ShrinkVisibility]: " << available_memeory << "GB" << std::endl;

    float used_memory = 0.0f;
    for (auto & image : images) {
        used_memory += image.GetWidth() * image.GetHeight();
    }
    const float G_byte = 1024 * 1024 * 1024;
    used_memory = 1.2 * used_memory / G_byte;

    std::cout << "Estimated Memory Consumption: " << used_memory << "GB" << std::endl;

    if (max_ram > 0) {
        available_memeory = std::min(max_ram, available_memeory);
    }
    std::cout << "Available Memory[Clamp]: " << available_memeory << "GB" << std::endl;

    if (available_memeory <= 1e-6) {
        ExceptionHandler(LIMITED_CPU_MEMORY, 
            JoinPaths(GetParentDir(workspace_path), "errors/dense"), "DenseSemantic").Dump();
        exit(StateCode::LIMITED_CPU_MEMORY);
    }

    size_t block_size = used_memory / available_memeory + 1;
    const size_t max_num_image_per_block = images.size() / block_size;

    std::cout << "Max image number of per block: " << max_num_image_per_block << std::endl;

#if 1
    std::vector<std::vector<size_t> > adj_vertices_per_vertex(num_vert);
    for (const auto & facet : mesh.faces_) {
        const int i0 = facet(0);
        const int i1 = facet(1);
        const int i2 = facet(2);
        auto & adj_verts0 = adj_vertices_per_vertex.at(i0);
        auto & adj_verts1 = adj_vertices_per_vertex.at(i1);
        auto & adj_verts2 = adj_vertices_per_vertex.at(i2);
        bool find = false;
        find = std::find_if(adj_verts0.begin(), adj_verts0.end(), [&](const int id) { return id == i1; }) != adj_verts0.end();
        if (!find) {
            if (adj_verts0.empty()) adj_verts0.push_back(i1);
            else adj_verts0.insert(std::lower_bound(adj_verts0.begin(), adj_verts0.end(), i1), i1);
            if (adj_verts1.empty()) adj_verts1.push_back(i0);
            else adj_verts1.insert(std::lower_bound(adj_verts1.begin(), adj_verts1.end(), i0), i0);
        }
        find = std::find_if(adj_verts0.begin(), adj_verts0.end(), [&](const int id) { return id == i2; }) != adj_verts0.end();
        if (!find) {
            if (adj_verts0.empty()) adj_verts0.push_back(i2);
            else adj_verts0.insert(std::lower_bound(adj_verts0.begin(), adj_verts0.end(), i2), i2);
            if (adj_verts2.empty()) adj_verts2.push_back(i0);
            else adj_verts2.insert(std::lower_bound(adj_verts2.begin(), adj_verts2.end(), i0), i0);
        }
        find = std::find_if(adj_verts1.begin(), adj_verts1.end(), [&](const int id) { return id == i2; }) != adj_verts1.end();
        if (!find) {
            if (adj_verts1.empty()) adj_verts1.push_back(i2);
            else adj_verts1.insert(std::lower_bound(adj_verts1.begin(), adj_verts1.end(), i2), i2);
            if (adj_verts2.empty()) adj_verts2.push_back(i1);
            else adj_verts2.insert(std::lower_bound(adj_verts2.begin(), adj_verts2.end(), i1), i1);
        }
    }

    std::vector<std::unordered_set<image_t> > block_images(block_size);
    std::vector<int> image_to_block_id(images.size(), -1);
    std::vector<bool> visited(num_vert, false);
    int block_id = 0;
    for (size_t i = 0; i < num_vert; ++i) {
        if (visited[i] || verts_vis[i].empty()) continue;
        
        if (block_images[block_id].size() > max_num_image_per_block) {
            block_id++;
            block_images.resize(block_id + 1);
        }

        std::queue<uint32_t> Q;
        Q.push(i);
        visited[i] = true;
        for (auto vis : verts_vis[i]) {
            if (image_to_block_id[vis] == -1) {
                image_to_block_id[vis] = block_id;
                block_images[block_id].insert(vis);
            }
        }
        while(!Q.empty()) {
            uint32_t cid = Q.front();
            Q.pop();
            const auto & adj_verts = adj_vertices_per_vertex.at(cid);
            for (auto nid : adj_verts) {
                auto & n_viss = verts_vis[nid];
                if (visited[nid] || n_viss.empty()) {
                    continue;
                }
                if (block_images[block_id].size() > max_num_image_per_block) {
                    break;
                }
                for (auto vis : n_viss) {
                    if (image_to_block_id[vis] == -1) {
                        image_to_block_id[vis] = block_id;
                        block_images[block_id].insert(vis);
                    }
                }
                Q.push(nid);
                visited[nid] = true;
            }

            if (block_images[block_id].size() > max_num_image_per_block) {
                break;
            }
        }
    }

    block_size = block_id + 1;
    std::cout << "Blocks: " << block_size << std::endl;

    std::vector<std::vector<size_t> > block_vertices(block_size);
    for (int block_id = 0; block_id < block_size; ++block_id) {
        block_vertices[block_id].reserve(images.size() / block_size);
    }
    for (size_t i = 0; i < num_vert; ++i) {
        std::unordered_set<image_t> block_ids;
        for (auto vis : verts_vis[i]) {
            if (image_to_block_id[vis] == -1) continue;
            block_ids.insert(image_to_block_id[vis]);
        }
        for (auto id : block_ids) {
            block_vertices[id].push_back(i);
        }
    }
    // for (int block_id = 0; block_id < block_size; ++block_id) {
    //     block_vertices[block_id].shrink_to_fit();
    // }
    block_vertices.shrink_to_fit();

    GetAvailableMemory(available_memeory);
    std::cout << "Available Memory: " << available_memeory << "GB" << std::endl;

    std::vector<std::vector<size_t> >().swap(adj_vertices_per_vertex);
    std::vector<int>().swap(image_to_block_id);
    std::vector<bool>().swap(visited);
    std::cout << "malloc_trim: " << malloc_trim(0) << std::endl;

    GetAvailableMemory(available_memeory);
    std::cout << "Available Memory[MemoryShrink]: " << available_memeory << "GB" << std::endl;

    std::vector<std::unique_ptr<Bitmap> > semantic_maps(images.size());
    std::vector<std::unordered_map<uint8_t, int> > label_voting(num_vert);
    for (int block_id = 0; block_id < block_size; ++block_id) {
        const auto & vertices = block_vertices[block_id];
        std::vector<image_t> image_ids;
        image_ids.insert(image_ids.end(), block_images[block_id].begin(), block_images[block_id].end());

        float used_memory = 0.0f;
        for (auto image_id : image_ids) {
            used_memory += images[image_id].GetWidth() * images[image_id].GetHeight();
        }
        used_memory = 1.2 * used_memory / G_byte;

        GetAvailableMemory(available_memeory);
        std::cout << StringPrintf("\rprocess block#%d, %d vertices, %d images, Available/Used Memory: %.2f/%.2fGB\n", 
                                block_id, vertices.size(), image_ids.size(), available_memeory, used_memory);

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < image_ids.size(); ++i) {
            int image_idx = image_ids[i];
            if (!semantic_maps[image_idx]) {
                const std::string image_name = image_names.at(image_idx);
                const std::string semantic_path = JoinPaths(semantic_maps_path, image_name);
                const std::string semantic_path_jpg = semantic_path.substr(0, semantic_path.size() - 3) + "png";
                if (ExistsFile(semantic_path_jpg)) {
                    semantic_maps[image_idx] = std::unique_ptr<Bitmap>(new Bitmap);
                    semantic_maps[image_idx]->Read(semantic_path_jpg, false);
                }
            }
            std::cout<< "\rPoints weighted [" << i << " / " << image_ids.size()<< "]" << std::flush;
        }
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < vertices.size(); ++i) {
            size_t vid = vertices[i];
            Eigen::Vector3f vert = mesh.vertices_.at(vid).cast<float>();
            for (auto cam_id : verts_vis[vid]) {
                if (!semantic_maps[cam_id]) continue;
                const mvs::Image& image = images[cam_id];
                Eigen::Map<const Eigen::RowMatrix3f> K(image.GetK());
                Eigen::Map<const Eigen::RowMatrix3f> R(image.GetR());
                Eigen::Map<const Eigen::Vector3f> T(image.GetT());

                Eigen::Vector3f proj = K * (R * vert + T);
                int u = proj[0] / proj[2];
                int v = proj[1] / proj[2];
                if (u < 0 || u >= image.GetWidth() || 
                    v < 0 || v >= image.GetHeight()) {
                    continue;
                }
                BitmapColor<uint8_t> semantic;
                semantic_maps[cam_id]->GetPixel(u, v, &semantic);
                uint8_t sid = semantic.r;
                if (sid == LABEL_PEDESTRIAN) {
                    continue;
                }
                label_voting[vid][sid]++;
            }
        }
        for (int i = 0; i < image_ids.size(); ++i) {
            int image_idx = image_ids[i];
            if (semantic_maps[image_idx]) {
                semantic_maps[image_idx].reset();
            }
        }
    }

    std::cout << "Semantic Label Voting" << std::endl;
    Timer vote_timer;
    vote_timer.Start();
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_vert; ++i) {
        int max_samples = 0;
        int best_label = -1;
        for (auto vote : label_voting[i]) {
            if (max_samples < vote.second) {
                max_samples = vote.second;
                best_label = vote.first;
            }
        }
        sem_mesh.vertex_labels_[i] = best_label;
    }
    vote_timer.PrintSeconds();

#else
    // std::vector<uint32_t> vertex_to_super_idx;
    std::unordered_map<uint32_t, std::vector<uint32_t> > super_idx_to_vectices;
    // std::vector<std::unordered_set<uint32_t> > neighbors_per_super;
    std::vector<std::vector<uint32_t> > super_visibilities;

    BuildSuperPixels(mesh, verts_vis, /*vertex_to_super_idx,*/ super_idx_to_vectices,
                     /*neighbors_per_super, */super_visibilities);

    struct Node {
        size_t i;
        std::vector<uint32_t>* visibilities;
    };
    std::vector<Node> sorted_viss(super_visibilities.size());
// #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < sorted_viss.size(); ++i) {
        auto & node = sorted_viss[i];
        node.i = i;
        node.visibilities = &super_visibilities.at(i);
    }

    std::sort(sorted_viss.begin(), sorted_viss.end(), 
        [&](const Node& a, const Node& b) {
            size_t min_size = std::min(a.visibilities->size(), b.visibilities->size());
            for (size_t i = 0; i < min_size; ++i) {
                if ((*a.visibilities)[i] < (*b.visibilities)[i]) return true;
                else if ((*a.visibilities)[i] > (*b.visibilities)[i]) return false;
            }
            return a.visibilities->size() < b.visibilities->size();
        });

    GetAvailableMemory(available_memeory);
    std::cout << "Available Memory[BuildSuperPixels]: " << available_memeory << "GB" << std::endl;

    std::vector<std::unordered_set<image_t> > block_images(block_size);
    std::vector<std::vector<size_t> > block_vertices(block_size);
    size_t block_id = 0;
    for (const auto & node : sorted_viss) {
        const int super_id = node.i;
        if (block_images[block_id].size() >= max_num_image_per_block && 
            max_num_image_per_block < images.size()) {
            block_id++;
            if (block_id >= block_size) {
                block_images.resize(block_id + 1);
                block_vertices.resize(block_id + 1);
            }
        }
        const auto & viss = super_visibilities[super_id];
        block_images[block_id].insert(viss.begin(), viss.end());
        block_vertices[block_id].insert(block_vertices[block_id].end(), super_idx_to_vectices[super_id].begin(), 
                                        super_idx_to_vectices[super_id].end());
    }

    GetAvailableMemory(available_memeory);
    std::cout << "Available Memory[ConstructBlocks]: " << available_memeory << "GB" << std::endl;

    for (auto & block : block_vertices) {
        block.shrink_to_fit();
    }
    block_vertices.shrink_to_fit();
    super_idx_to_vectices.clear();
    std::vector<Node>().swap(sorted_viss);
    sorted_viss.shrink_to_fit();
    std::vector<std::vector<uint32_t> >().swap(super_visibilities);
    super_visibilities.shrink_to_fit();
    std::cout << "malloc_trim: " << malloc_trim(0) << std::endl;

    GetAvailableMemory(available_memeory);
    std::cout << "Available Memory[ReleaseBlocks]: " << available_memeory << "GB" << std::endl;

    std::cout << "Blocks: " << block_images.size() << std::endl;
    
    std::vector<int> num_reference(images.size(), 0);
    for (int i = 0; i < block_images.size(); ++i) {
        const auto & image_ids = block_images.at(i);
        for (auto image_idx : image_ids) {
            num_reference[image_idx]++;
        }
        std::cout << StringPrintf("Block#%d: %d images, %d vertices\n", i, image_ids.size(), block_vertices[i].size());
    }

    std::vector<std::unique_ptr<Bitmap> > semantic_maps(images.size());
    for (int i = 0; i < block_vertices.size(); ++i) {
        const std::vector<size_t>& vertex_ids = block_vertices.at(i);
        std::vector<image_t> image_ids;
        image_ids.insert(image_ids.end(), block_images[i].begin(), block_images[i].end());
        int num_load_map = 0;
        for (auto image_idx : image_ids) {
            if (!semantic_maps[image_idx]) {
                num_load_map++;
            }
        }

#pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < image_ids.size(); ++j) {
            int image_idx = image_ids[j];
            if (!semantic_maps[image_idx]) {
                const std::string image_name = image_names.at(image_idx);
                const std::string semantic_path = JoinPaths(semantic_maps_path, image_name);
                const std::string semantic_path_jpg = semantic_path.substr(0, semantic_path.size() - 3) + "png";
                if (ExistsFile(semantic_path_jpg)) {
                    semantic_maps[image_idx] = std::unique_ptr<Bitmap>(new Bitmap);
                    semantic_maps[image_idx]->Read(semantic_path_jpg, false);
                }
            }
        }

#pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < vertex_ids.size(); ++j) {
            Eigen::Vector3f vert = mesh.vertices_.at(vertex_ids[j]).cast<float>();

            float max_samples = 0;
            int best_label = -1;
            std::vector<int> samps_per_label(256, 0);
            for (auto cam_id : verts_vis[vertex_ids[j]]) {
                const mvs::Image& image = images.at(cam_id);
                Eigen::Map<const Eigen::RowMatrix3f> K(image.GetK());
                Eigen::Map<const Eigen::RowMatrix3f> R(image.GetR());
                Eigen::Map<const Eigen::Vector3f> T(image.GetT());

                Eigen::Vector3f proj = K * (R * vert + T);
                int u = proj[0] / proj[2];
                int v = proj[1] / proj[2];
                if (u < 0 || u >= image.GetWidth() || 
                    v < 0 || v >= image.GetHeight()) {
                    continue;
                }
                auto & semantic_map = semantic_maps.at(cam_id);
                if (semantic_map && semantic_map->NumBytes() > 0) {
                    BitmapColor<uint8_t> semantic;
                    semantic_maps.at(cam_id)->GetPixel(u, v, &semantic);
                    uint8_t sid = semantic.r;
                    if (sid == LABEL_PEDESTRIAN) {
                        continue;
                    }
                    samps_per_label[sid]++;
                    if (samps_per_label[sid] > max_samples) {
                        max_samples = samps_per_label[sid];
                        best_label = sid;
                    }
                }
            }
            sem_mesh.vertex_labels_[vertex_ids[j]] = best_label;
            std::vector<uint32_t>().swap(verts_vis[vertex_ids[j]]);
        }
        for (auto image_idx : image_ids) {
            num_reference[image_idx]--;
            if (num_reference[image_idx] == 0) {
                semantic_maps.at(image_idx).reset();
            }
        }
        int num_resident_map = 0;
        for (int j = 0; j < images.size(); ++j) {
            if (num_reference[j] != 0) num_resident_map++;
        }
        std::cout << StringPrintf("\rprocess block#%d(%d), load/resident(%d/%d)\n", i, vertex_ids.size(), num_load_map, num_resident_map);
        GetAvailableMemory(available_memeory);
        std::cout << "Available Memory: " << available_memeory << "GB" << std::endl;
    }
#endif
    // for (size_t i = 0; i < num_vert; ++i) {
    //     int best_label = vert_labels[i];
    //     sem_mesh.vertex_labels_.emplace_back(best_label);
    // }
    // RefineSemantization(sem_mesh);
    sem_mesh.vertex_colors_.reserve(num_vert);
    for (size_t i = 0; i < num_vert; ++i) {
        int best_label = sem_mesh.vertex_labels_[i];
        best_label = (best_label + 256) % 256;
        Eigen::Vector3d rgb;
        rgb[0] = adepallete[best_label * 3];
        rgb[1] = adepallete[best_label * 3 + 1];
        rgb[2] = adepallete[best_label * 3 + 2];
        sem_mesh.vertex_colors_.emplace_back(rgb);
    }
}

void ModelHoleFill(TriangleMesh& mesh, TriangleMesh& sem_mesh, 
    const std::vector<uint8_t>& removal_labels, 
    const int num_isolated_pieces, const float border_update_ratio) {
    if (mesh.faces_.size() == 0) {
        return;
    }

    std::unordered_set<uint8_t> removal_labels_map;
    std::for_each(removal_labels.begin(), removal_labels.end(), 
    [&](uint8_t label) {
        removal_labels_map.insert(label);
    });

    std::unordered_set<int> removal_vtx_map;

    mesh.vertex_status_.resize(mesh.vertices_.size());
    std::fill(mesh.vertex_status_.begin(), mesh.vertex_status_.end(), 
              VERTEX_STATUS::NORMAL);

    size_t i, j, vtx_index = 0;
    for (i = 0, j = 0; i < mesh.faces_.size(); ++i) {
        auto & facet = mesh.faces_.at(i);
        int8_t label0 = sem_mesh.vertex_labels_.at(facet[0]);
        int8_t label1 = sem_mesh.vertex_labels_.at(facet[1]);
        int8_t label2 = sem_mesh.vertex_labels_.at(facet[2]);
        if (removal_labels_map.count(label0) != 0 ||
            removal_labels_map.count(label1) != 0 ||
            removal_labels_map.count(label2) != 0) {
            removal_vtx_map.insert(facet[0]);
            removal_vtx_map.insert(facet[1]);
            removal_vtx_map.insert(facet[2]);
            continue;
        }
        mesh.vertex_status_.at(facet[0]) = VERTEX_STATUS::UPDATE;
        mesh.vertex_status_.at(facet[1]) = VERTEX_STATUS::UPDATE;
        mesh.vertex_status_.at(facet[2]) = VERTEX_STATUS::UPDATE;
        mesh.faces_.at(j) = facet;
        j = j + 1;
    }
    mesh.faces_.resize(j);
    for (i = 0; i < mesh.vertices_.size(); ++i) {
        if (removal_vtx_map.find(i) == removal_vtx_map.end()) {
            mesh.vertex_status_.at(i) = VERTEX_STATUS::NORMAL;
        }
    }

    mesh.RemoveIsolatedPieces(num_isolated_pieces);

    MeshInfo mesh_info;
    mesh_info.initialize(mesh);
    mesh_info.resolve_complex_vertex(mesh);

    mesh.RemoveIsolatedPieces(num_isolated_pieces);
    mesh_info.clear();
    mesh_info.initialize(mesh);

    // WriteTriangleMeshObj("model-trim.obj", mesh);

    std::vector<std::vector<int> > lists;
    mesh_info.get_border_lists(lists);

    for (i = 0, j = 0; i < lists.size(); ++i) {
        auto & list = lists.at(i);
        int num_border_vtx = 0;
        for (auto & vtx_id : list) {
            if (mesh.vertex_status_.at(vtx_id) == VERTEX_STATUS::UPDATE) {
                num_border_vtx++;
            }
        }
        float ratio = num_border_vtx * 1.0f / list.size();
        if (ratio > border_update_ratio) {
            lists.at(j) = list;
            j = j + 1;
        }
    }
    lists.resize(j);

    mesh.HollFill(lists);
    // WriteTriangleMeshObj("model-fill.obj", mesh);
}

void Fix(TriangleMesh& mesh) {
    const int invalid_label = -1;
    std::unordered_map<int, std::vector<size_t> > adj_facets_per_vert;

    auto FixNonConsistentLabel = [&]() {
        adj_facets_per_vert.clear();
        for (size_t i = 0; i < mesh.faces_.size(); ++i) {
            auto facet = mesh.faces_[i];
            if (mesh.vertex_labels_[facet[0]] == invalid_label) {
                adj_facets_per_vert[facet[0]].push_back(i);
            }
            if (mesh.vertex_labels_[facet[1]] == invalid_label) {
                adj_facets_per_vert[facet[1]].push_back(i);
            }
            if (mesh.vertex_labels_[facet[2]] == invalid_label) {
                adj_facets_per_vert[facet[2]].push_back(i);
            }
        }

        auto vertex_labels = mesh.vertex_labels_;
        auto vertex_colors = mesh.vertex_colors_;
        for (auto adj_facets : adj_facets_per_vert) {
            std::map<int, int> adj_labels;
            for (auto f_idx : adj_facets.second) {
                auto facet = mesh.faces_[f_idx];
                int label;
                if (facet[0] != adj_facets.first) {
                    label = mesh.vertex_labels_[facet[0]];
                    adj_labels[label]++;
                }
                if (facet[1] != adj_facets.first) {
                    label = mesh.vertex_labels_[facet[1]];
                    adj_labels[label]++;
                }
                if (facet[2] != adj_facets.first) {
                    label = mesh.vertex_labels_[facet[2]];
                    adj_labels[label]++;
                }
            }

            auto best_label = adj_labels.rbegin();
            if (best_label->first != invalid_label) {
                vertex_labels[adj_facets.first] = best_label->first;
                Eigen::Vector3d rgb;
                rgb[0] = adepallete[best_label->first * 3];
                rgb[1] = adepallete[best_label->first * 3 + 1];
                rgb[2] = adepallete[best_label->first * 3 + 2];
                vertex_colors[adj_facets.first] = rgb;
            }
        }
        std::swap(mesh.vertex_colors_, vertex_colors);
        std::swap(mesh.vertex_labels_, vertex_labels);

        return adj_facets_per_vert.size();
    };

    int i, iter = 0;
    while((i = FixNonConsistentLabel()) && iter < 3) {
        iter++;
        std::cout << StringPrintf("Fix %d / %d vertex labels",
            i, mesh.vertex_labels_.size()) << std::endl;
    }
}

std::string configuration_file_path;

int main(int argc, char *argv[]) {

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");    
    PrintHeading(std::string("Version: ") + __VERSION__);

    configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
    std::string image_type = param.GetArgument("image_type", "perspective");
    
    ParamOption::min_consistent_facet = param.GetArgument("min_consistent_facet", 1000);
    ParamOption::dist_point_to_line = param.GetArgument("dist_point_to_line", 0.f);
    ParamOption::angle_diff_thres = param.GetArgument("angle_diff", 20.0f);
    ParamOption::dist_ratio_point_to_plane = param.GetArgument("dist_ratio_point_to_plane", 10.0f);
    ParamOption::ratio_singlevalue_xz = param.GetArgument("ratio_singlevalue_xz", 1500.0f);
    ParamOption::ratio_singlevalue_yz = param.GetArgument("ratio_singlevalue_yz", 400.0f);

    int num_isolated_pieces = param.GetArgument("num_isolated_pieces", 1000);
    float border_update_ratio = param.GetArgument("border_update_ratio", 0.99f);

    std::string black_list_file_path = param.GetArgument("black_list_file_path", "");

    std::string semantic_table_path = param.GetArgument("semantic_table_path", "");
    if (ExistsFile(semantic_table_path)) {
        LoadSemanticColorTable(semantic_table_path.c_str());
    }

    float max_ram = param.GetArgument("max_ram", -1.f);
    std::cout << "max ram: " << max_ram << "GB" << std::endl;

    const int max_num_facets = 20000000;

    std::vector<uint8_t> removal_labels;
    if (ExistsFile(black_list_file_path)) {
        LoadSemanticLabels(black_list_file_path, removal_labels);
    }

    int num_reconstruction = 0;
    for (size_t reconstruction_idx = 0; ; reconstruction_idx++) {

        Timer timer;
        timer.Start();

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

        auto undistort_image_path = 
            JoinPaths(dense_reconstruction_path, IMAGES_DIR);
        auto semantic_maps_path = 
            JoinPaths(dense_reconstruction_path, SEMANTICS_DIR);

        std::vector<mvs::Image> images;
        std::vector<std::string> image_names;
        if (image_type.compare("perspective") == 0 || image_type.compare("rgbd") == 0) {
            Workspace::Options workspace_options;
            workspace_options.max_image_size = -1;
            workspace_options.image_as_rgb = false;
            workspace_options.image_path = undistort_image_path;
            workspace_options.workspace_path = dense_reconstruction_path;
            workspace_options.workspace_format = image_type;

            Workspace workspace(workspace_options);
            const Model& model = workspace.GetModel();
            images = model.images;
            for (size_t i = 0; i < images.size(); ++i) {
                image_names.push_back(model.GetImageName(i));
            }
        } else if (image_type.compare("panorama") == 0) {
            std::vector<image_t> image_ids;
            std::vector<std::vector<int> > overlapping_images;
            std::vector<std::pair<float, float> > depth_ranges;
            if (!ImportPanoramaWorkspace(dense_reconstruction_path, image_names,
                images, image_ids, overlapping_images, depth_ranges, false)) {
                return 1;
            }
        }

        auto in_model_path = JoinPaths(dense_reconstruction_path, MODEL_NAME);
        auto sem_model_name = in_model_path.substr(0, in_model_path.size() - 4) + "_sem.obj";

        TriangleMesh mesh, sem_mesh;

        if (!ExistsFile(sem_model_name)) {
            ReadTriangleMeshObj(in_model_path, mesh, true, false);

            // if (mesh.faces_.size() > max_num_facets) {
            //     double fDecimate = 0.25;
            //     std::cout << "fDecimate: " << fDecimate << std::endl;
            //     mesh.Clean(fDecimate, 0, false, 0, 0, false);
            //     mesh.ComputeNormals();
            // }

            sem_mesh = mesh;
            ModelSemantization(images, image_names, workspace_path, semantic_maps_path, mesh, sem_mesh, max_ram);

            Fix(sem_mesh);

            std::cout << sem_model_name << std::endl;
            WriteTriangleMeshObj(sem_model_name, sem_mesh, true, true);
        } else {
            ReadTriangleMeshObj(in_model_path, mesh, true, false);
            ReadTriangleMeshObj(sem_model_name, sem_mesh, true, true);
        }
        
        // if (removal_labels.size() > 0) {

        //     std::cout << "Remove Semantic Labels: ";
        //     for (auto label : removal_labels) {
        //         std::cout << (int)label << " ";
        //     }
        //     std::cout << std::endl;

        //     ModelHoleFill(mesh, sem_mesh, removal_labels, num_isolated_pieces, border_update_ratio);

        //     auto fill_model_name = in_model_path.substr(0, in_model_path.size() - 4) + "_fill.obj";
        //     WriteTriangleMeshObj(fill_model_name, mesh);
        // }

        timer.PrintMinutes();

        num_reconstruction++;
    }
    return 0;
}
