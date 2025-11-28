#include "util/semantic_table.h"
#include "util/exception_handler.h"
#include "mvs/delaunay_meshing.h"

#include <malloc.h>

#define EPSILON std::numeric_limits<float>::epsilon()
#define MAX_INT std::numeric_limits<int>::max()

// Types
typedef CGAL::Exact_predicates_inexact_constructions_kernel kernel_t;
typedef kernel_t::FT FT;
typedef kernel_t::Point_3 point_3_t;
typedef CGAL::Search_traits_3<kernel_t> tree_traits_3_t;
typedef CGAL::Orthogonal_k_neighbor_search<tree_traits_3_t> neighbor_search_3_t;
typedef neighbor_search_3_t::iterator search_3_iterator_t;
typedef neighbor_search_3_t::Tree tree_3_t;
typedef boost::tuple<int, point_3_t> indexed_point_3_tuple_t;

namespace sensemap {
namespace mvs {
void LoadLidarPose(const std::string file_path, std::vector<mvs::Image>& lidars){
    std::ifstream file(file_path, std::ios::binary);
    CHECK(file.is_open()) << file_path;

    uint64_t num_register_sweep = 0;
    num_register_sweep = ReadBinaryLittleEndian<uint64_t>(&file);
    // lidars.reserve(num_register_sweep);
    lidars.resize(num_register_sweep);

    for (int i = 0; i < num_register_sweep; ++i) {
        sweep_t sweep_id = ReadBinaryLittleEndian<sweep_t>(&file);

        Eigen::Vector4d qvec;
        qvec[0] = ReadBinaryLittleEndian<double>(&file);
        qvec[1] = ReadBinaryLittleEndian<double>(&file);
        qvec[2] = ReadBinaryLittleEndian<double>(&file);
        qvec[3] = ReadBinaryLittleEndian<double>(&file);

        Eigen::Vector3d tvec;
        tvec[0] = ReadBinaryLittleEndian<double>(&file);
        tvec[1] = ReadBinaryLittleEndian<double>(&file);
        tvec[2] = ReadBinaryLittleEndian<double>(&file);
        
        std::string sweep_name;
        char name_char;
        do {
            file.read(&name_char, 1);
            if (name_char != '\0') {
                sweep_name += name_char;
                // lidarsweep.Name() += name_char;
            }
        } while (name_char != '\0');


        const Eigen::RowMatrix3f K = Eigen::RowMatrix3f::Identity();
        const Eigen::RowMatrix3f R = 
        QuaternionToRotationMatrix(qvec).cast<float>();
        const Eigen::Vector3f T = tvec.cast<float>();

        std::string sweep_path = sweep_name;
        
        lidars.at(sweep_id) = mvs::Image(sweep_path, 1, 1, 
            K.data(), R.data(), T.data(), false, true);
    }
    file.close();

    return;
}

void ColorMap(const float intensity, uint8_t &r, uint8_t &g, uint8_t &b){
    if(intensity <= 0.25) {
        r = 0;
        g = 0;
        b = (intensity) * 4 * 255;
        return;
    }

    if(intensity <= 0.5) {
        r = 0;
        g = (intensity - 0.25) * 4 * 255;
        b = 255;
        return;
    }

    if(intensity <= 0.75) {
        r = 0;
        g = 255;
        b = (0.75 - intensity) * 4 * 255;
        return;
    }

    if(intensity <= 1.0) {
        r = (intensity - 0.75) * 4 * 255;
        g = (1.0 - intensity) * 4 * 255;
        b = 0;
        return;
    }
};

void prepare_mesh(MeshInfo * mesh_info, TriangleMesh mesh) {

    /* Update vertex infos. */
    mesh_info->clear();
    mesh_info->initialize(mesh);

}

void NonUniformSampling(std::vector<PlyPoint> &points,
                        std::vector<std::vector<uint32_t> > &points_visibility,
                        const std::vector<mvs::Image> &images) {
    std::cout << "Non-uniform sampling points" << std::endl;
    const int win_r = 5;
    const float dist_thres = 1e-3;

    std::vector<unsigned char> sample_flags(points.size(), 1);

    for (size_t image_idx = 0; image_idx < images.size(); ++image_idx) {
        const mvs::Image &image = images.at(image_idx);
        const int width = image.GetWidth();
        const int height = image.GetWidth();
        const float *P = image.GetP();

        std::vector<std::vector<int> > grid_to_pt(height, std::vector<int>(width, -1));

        for (size_t i = 0; i < points.size(); ++i) {
            if (!sample_flags[i]) {
                continue;
            }
            PlyPoint &pt = points.at(i);
            std::vector<uint32_t>& vis_ids = points_visibility[i];
            auto res = std::find(vis_ids.begin(), vis_ids.end(), image_idx);
            if (res == vis_ids.end()) {
                continue;
            }
            float m = P[8] * pt.x + P[9] * pt.y + P[10] * pt.z + P[11];
            int u = (P[0] * pt.x + P[1] * pt.y + P[2] * pt.z + P[3]) / m + 0.5;
            int v = (P[4] * pt.x + P[5] * pt.y + P[6] * pt.z + P[7]) / m + 0.5;
            if (u < 0 || u >= width || v < 0 || v >= height) {
                continue;
            }
            grid_to_pt[v][u] = i;
        }
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                int point_idx = grid_to_pt[r][c];
                if (point_idx == -1 || !sample_flags[point_idx]) {
                    continue;
                }

                Eigen::Vector3f pt(&points.at(point_idx).x);
                Eigen::Vector3f normal(&points.at(point_idx).nx);

                int sx = std::max(c - win_r, 0);
                int ex = std::min(c + win_r, width - 1);
                int sy = std::max(r - win_r, 0);
                int ey = std::min(r + win_r, height - 1);
                for (int wr = sy; wr <= ey; ++wr) {
                    for (int wc = sx; wc <= ex; ++wc) {
                        if (wr == r && wc == c) {
                            continue;
                        }
                        int neighbor_idx = grid_to_pt[wr][wc];
                        if (neighbor_idx == -1 || !sample_flags[point_idx]) {
                            continue;
                        }
                        Eigen::Vector3f n_pt(&points.at(neighbor_idx).x);
                        float dist = std::fabs((n_pt - pt).dot(normal));
                        if (dist < dist_thres) {
                            sample_flags[neighbor_idx] = 0;
                        }
                    }
                }
            }
        }
        std::cout << StringPrintf("\rSampling Frame#%d", image_idx) << std::flush;
    }
    std::cout << std::endl;

    size_t i, j;
    for (i = 0, j = 0; i < points.size(); ++i) {
        if (sample_flags[i]) {
            points[j] = points[i];
            points_visibility[j] = points_visibility[i];
            j = j + 1;
        }
    }
    points.resize(j);
    points_visibility.resize(j);
}
namespace ParamOption {
int min_consistent_facet = 200;
double dist_point_to_line = 0.0;
double angle_diff_thres = 20.0;
double dist_ratio_point_to_plane = 10.0;
double ratio_singlevalue_xz = 1500.0;
double ratio_singlevalue_yz = 400.0;
};

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

bool DelaMeshing::Options::Check() const {
  CHECK_OPTION_GE(plane_insert_factor, 2.0f);
  CHECK_OPTION_GT(plane_score_thred, 0.0f);
  CHECK_OPTION_LE(plane_score_thred, 1.0f);
  CHECK_OPTION_GE(num_isolated_pieces, 0);
  CHECK_OPTION_GE(decimate_mesh, 0);
//   CHECK_OPTION_GT(remove_spurious, 0);
  CHECK_OPTION_GT(ram_eff_factor, 0);
  CHECK_OPTION_GT(overlap_factor, 0);
  return true;
}

void DelaMeshing::Options::Print() const {
  PrintHeading2("DelaMeshing::Options");
  PrintOption(dist_insert);
  PrintOption(diff_depth);
  PrintOption(sigma);
  PrintOption(decimate_mesh);
  PrintOption(only_remove_edge_spurious);
  PrintOption(remove_spurious);
  PrintOption(remove_spikes);
  PrintOption(close_holes);
  PrintOption(smooth_mesh);
  PrintOption(num_isolated_pieces);
  PrintOption(fix_mesh);
  PrintOption(adaptive_insert);
  PrintOption(plane_insert_factor);
  PrintOption(plane_score_thred);
  PrintOption(roi_mesh);
  PrintOption(roi_box_width);
  PrintOption(roi_box_factor);
  PrintOption(overlap_factor);
  PrintOption(max_ram);
  PrintOption(ram_eff_factor);
}

DelaMeshing::DelaMeshing(const Options& options,
                        const std::string& workspace_path,
                        const std::string& image_type,
                        const int reconstrction_idx,
                        const int cluster_idx)
    : num_cluster_(0),
      options_(options),
      workspace_path_(workspace_path),
      image_type_(image_type),
      select_reconstruction_idx_(reconstrction_idx),
      select_cluster_idx_(cluster_idx) {
  CHECK(options_.Check());
}

void DelaMeshing::ReadWorkspace() {
  for (size_t cluster_idx = 0; ;cluster_idx++) {
    const auto& reconstruction_path = 
        JoinPaths(workspace_path_, std::to_string(select_reconstruction_idx_));
    const auto& cluster_reconstruction_path = 
        JoinPaths(reconstruction_path, std::to_string(cluster_idx));
    if (!ExistsDir(cluster_reconstruction_path) && cluster_idx != 0) {
        break;
    }

    num_cluster_++;
  }
  std::cout << "Reading workspace (" << num_cluster_ << " cluster)..." << std::endl;
}

void DelaMeshing::Run() {
  options_.Print();
  ReadWorkspace();
  std::cout << std::endl;

  if (IsStopped()) {
    GetTimer().PrintMinutes();
    return;
  }

  size_t reconstruction_idx = select_reconstruction_idx_;
  auto reconstruction_path =
    JoinPaths(workspace_path_, std::to_string(reconstruction_idx));
  auto dense_reconstruction_path =
    JoinPaths(reconstruction_path, DENSE_DIR);
  if (!ExistsDir(dense_reconstruction_path)) {
    return;
  }
  auto undistort_image_path =
    JoinPaths(dense_reconstruction_path, IMAGES_DIR);

  if (ExistsFile(JoinPaths(dense_reconstruction_path, MODEL_NAME))){
    boost::filesystem::remove(JoinPaths(dense_reconstruction_path, MODEL_NAME));
    std::cout << "remove file: " << JoinPaths(dense_reconstruction_path, MODEL_NAME) << std::endl;
  }
  if (ExistsFile(JoinPaths(dense_reconstruction_path, SEM_MODEL_NAME))){
    boost::filesystem::remove(JoinPaths(dense_reconstruction_path, SEM_MODEL_NAME));
    std::cout << "remove file: " << JoinPaths(dense_reconstruction_path, SEM_MODEL_NAME) << std::endl;
  }

  Workspace::Options workspace_options;
  workspace_options.max_image_size = -1;
  workspace_options.image_as_rgb = false;
  workspace_options.image_path = undistort_image_path;
  workspace_options.workspace_path = dense_reconstruction_path;
  workspace_options.workspace_format = image_type_;
//   Workspace workspace(workspace_options);
  workspace_.reset(new Workspace(workspace_options));

  size_t cluster_begin = select_cluster_idx_ < 0 ? 0 : select_cluster_idx_;
  size_t cluster_end = select_cluster_idx_ < 0 ? num_cluster_ : select_cluster_idx_ + 1;
  for (size_t sfm_cluster_idx = cluster_begin; sfm_cluster_idx < cluster_end; sfm_cluster_idx++) {
    std::string cluster_rect_path = 
        JoinPaths(reconstruction_path, std::to_string(sfm_cluster_idx));
    if (!ExistsDir(cluster_rect_path)){
      if (sfm_cluster_idx != 0){
        break;
      }
      if (!ExistsFile(JoinPaths(dense_reconstruction_path, FUSION_NAME)) && 
          !ExistsFile(JoinPaths(dense_reconstruction_path, LIDAR_NAME))){
        break;
      }
      cluster_rect_path = dense_reconstruction_path;
    }

    PrintHeading1(StringPrintf("Cluster DelaMeshing# %d - %d", 
                  reconstruction_idx, sfm_cluster_idx));
    std::cout << "cluster_rect_path:" << cluster_rect_path << std::endl;

    DelaunayRecon(dense_reconstruction_path, cluster_rect_path, 
                  undistort_image_path, cluster_rect_path);

  }

  if (cluster_end == num_cluster_){
    MergeMeshing(reconstruction_path);
  }

  GetTimer().PrintMinutes();
}

void DelaMeshing::DelaunayRecon(const std::string &dense_reconstruction_path,
                                const std::string &cluster_rect_path,
                                const std::string &undistort_image_path,
                                const std::string &output_path) {
    auto output_obj_path = JoinPaths(output_path, MODEL_NAME);
    auto output_objsem_path = JoinPaths(output_path, SEM_MODEL_NAME);
    auto input_path = JoinPaths(cluster_rect_path, FUSION_NAME);
    auto input_vis_path = JoinPaths(cluster_rect_path, FUSION_NAME) + ".vis";
    auto input_sem_path = JoinPaths(cluster_rect_path, FUSION_NAME) + ".sem";
    auto input_wgt_path = JoinPaths(cluster_rect_path, FUSION_NAME) + ".wgt";
    auto input_sco_path = JoinPaths(cluster_rect_path, FUSION_NAME) + ".sco";

    auto cluster_roi_path = cluster_rect_path;
    if (!ExistsFile(JoinPaths(cluster_roi_path, ROI_BOX_NAME)) && 
       ExistsFile(JoinPaths(cluster_roi_path, "..", ROI_BOX_NAME))){
      cluster_roi_path = JoinPaths(cluster_roi_path, "..");
    }
    auto ori_box_path = JoinPaths(cluster_roi_path, ROI_BOX_NAME);
    
    auto lidar_bin_path = JoinPaths(dense_reconstruction_path, SPARSE_DIR, "lidars.bin");
    auto lidar_path = JoinPaths(dense_reconstruction_path, LIDAR_NAME);
    auto lidar_vis_path = JoinPaths(dense_reconstruction_path, LIDAR_NAME) + ".vis";
    auto lidar_wgt_path = JoinPaths(dense_reconstruction_path, LIDAR_NAME) + ".wgt";

    std::vector<PlyPoint> ply_points;
    std::vector<std::vector<uint32_t> > vis_points;
    std::vector<std::vector<float> >  weight_points;
    std::vector<float> score_points;
    bool write_as_rgb = true;
    bool write_as_sem = true;

    if (ExistsFile(input_path) && ExistsFile(input_vis_path)) {
        ply_points = ReadPly(input_path);
        ReadPointsVisibility(input_vis_path, vis_points);
        if (ExistsFile(input_sem_path)) {
            ReadPointsSemantic(input_sem_path, ply_points);
        }
        if(ExistsFile(input_wgt_path)){
            ReadPointsWeight(input_wgt_path, weight_points);
        } else {
            weight_points.resize(vis_points.size());
            for (size_t i = 0; i < vis_points.size(); i++){
                std::vector<float> temp_weight(vis_points[i].size(), 1.0f);
                weight_points[i].swap(temp_weight);
            }
        }

        if (ExistsFile(input_sco_path)){
            ReadPointsScore(input_sco_path, score_points);
        }
        std::cout << "Read fused.ply " << ply_points.size() << " points." << std::endl;
    }
    const bool has_score = !score_points.empty();

    std::vector<PlyPoint> lidar_points;
    std::vector<std::vector<uint32_t> > lidar_vis_points;
    std::vector<std::vector<float> > lidar_wgt_points;
    std::vector<float> lidar_sco_points;
    if (ExistsFile(lidar_path) && ExistsFile(lidar_vis_path)) {
        lidar_points = ReadPly(lidar_path);
        ReadPointsVisibility(lidar_vis_path, lidar_vis_points);
        ReadPointsWeight(lidar_wgt_path, lidar_wgt_points);
        if (has_score){
            std::vector<float> temp_sco(lidar_points.size(), 0.1f);
            lidar_sco_points.swap(temp_sco);
        }
        std::cout << "Read lidar.ply " << lidar_points.size() << " points." << std::endl;
    }

    if (ply_points.size() == 0 && lidar_points.size() == 0) {
        std::cerr << "Warning! No Dense Points(visual or lidar) generated!" << std::endl;
        ExceptionHandler(StateCode::POINT_CLOUD_IS_EMPTY, 
            JoinPaths(GetParentDir(workspace_path_), "errors/dense"), "DenseMeshing").Dump();
        exit(StateCode::POINT_CLOUD_IS_EMPTY);
        // return;
    }

    struct MeshBox roi_meshbox;
    if (options_.roi_mesh && ExistsFile(ori_box_path)){
        Box roi_box;
        ReadBoundBoxText(ori_box_path, roi_box);
        float roi_box_factor = options_.roi_box_factor;
        if (options_.roi_box_width < 0 && options_.roi_box_factor < 0){
            roi_box_factor = 0.01;
        }
        roi_box.SetBoundary(options_.roi_box_width * 2, roi_box_factor * 2);
        roi_meshbox = roi_box.ToMeshBox();
        roi_meshbox.SetBoundary();

        auto Select = [&](std::vector<PlyPoint>& pnts, 
            std::vector<std::vector<uint32_t> >& vis_pnts,
            std::vector<std::vector<float> >& wgt_pnts,
            std::vector<float>& sco_pnts){               
            std::vector<PlyPoint> select_points;
            std::vector<std::vector<uint32_t> > select_vis_points;
            std::vector<std::vector<float> >  select_weight_points;
            std::vector<float> select_score_points;
            select_points.reserve(pnts.size());
            select_vis_points.reserve(pnts.size());
            select_weight_points.reserve(pnts.size());
            if (has_score){
                select_score_points.reserve(pnts.size());
            }
            for (int i = 0; i < pnts.size(); i++){
                Eigen::Vector3f pnt(pnts.at(i).x, pnts.at(i).y, pnts.at(i).z);
                Eigen::Vector3f trans_pnt = roi_box.rot * pnt;
                if (trans_pnt.x() < roi_box.x_box_min || trans_pnt.x() > roi_box.x_box_max ||
                    trans_pnt.y() < roi_box.y_box_min || trans_pnt.y() > roi_box.y_box_max){
                    continue;
                }
                select_points.push_back(pnts.at(i));
                select_vis_points.push_back(vis_pnts.at(i));
                select_weight_points.push_back(wgt_pnts.at(i));
                if (has_score){
                    select_score_points.push_back(sco_pnts.at(i));
                }
            }
            pnts.swap(select_points);
            vis_pnts.swap(select_vis_points);
            wgt_pnts.swap(select_weight_points);
            if (has_score){
                sco_pnts.swap(select_score_points);
            }
        };
        std::cout << "Ori Size: " << ply_points.size() << ", " << lidar_points.size() << std::endl;
        Select(ply_points, vis_points, weight_points, score_points);
        if(!lidar_points.empty()){
            Select(lidar_points, lidar_vis_points, lidar_wgt_points, lidar_sco_points);
        }
        // std::string sample_path = JoinPaths(cluster_rect_path, "sampled_fused.ply");
        // std::string sample_vis_path = JoinPaths(cluster_rect_path, "sampled_fused.ply") + ".vis";
        // std::string sample_wgt_path = JoinPaths(cluster_rect_path, "sampled_fused.ply") + ".wgt";
        // WriteBinaryPlyPoints(sample_path, ply_points);
        // WritePointsVisibility(sample_vis_path, vis_points);
        // WritePointsWeight(sample_wgt_path, weight_points);
        std::cout << "Filter Size: " << ply_points.size() << ", " << lidar_points.size() << std::endl;
    }

    size_t num_vision_points = ply_points.size();
    size_t num_lidar_points = lidar_points.size();
    bool has_lidar = false;
    if (!lidar_points.empty()){
        ply_points.insert(ply_points.end(), lidar_points.begin(), lidar_points.end());
        vis_points.insert(vis_points.end(), lidar_vis_points.begin(), lidar_vis_points.end());
        weight_points.insert(weight_points.end(), lidar_wgt_points.begin(), lidar_wgt_points.end());
        if (has_score){
            score_points.insert(score_points.end(), lidar_sco_points.begin(), lidar_sco_points.end());
        }

        lidar_points.clear();
        lidar_vis_points.clear();
        lidar_wgt_points.clear();
        lidar_sco_points.clear();
        has_lidar = true;
    }
    malloc_trim(0);

    size_t num_ply_points = ply_points.size();

    if (ply_points.size() == 0){
        std::cerr << "Warning! No Dense Points Selected!" << std::endl;
        ExceptionHandler(StateCode::POINT_CLOUD_IS_EMPTY, 
            JoinPaths(GetParentDir(workspace_path_), "errors/dense"), "DenseMeshing").Dump();
        exit(StateCode::POINT_CLOUD_IS_EMPTY);
        // return;
    }

    TriangleMesh obj_mesh;
    std::vector<int > saved_cluster_ids;
#if 1
    if (image_type_.compare("perspective") == 0 ||
        image_type_.compare("rgbd") == 0) {

        // workspace_->SetModel(cluster_rect_path);
        const Model& model = workspace_->GetModel();
        std::size_t num_images = model.images.size();
        std::vector<mvs::Image> lidars_pose;
        if (has_lidar){
            LoadLidarPose(lidar_bin_path, lidars_pose);
            
            for(size_t idx = num_vision_points; idx < num_ply_points; idx++){
                for(auto& vis : vis_points.at(idx)){
                    vis += num_images;
                }
            }
        }
        if (0) {
            NonUniformSampling(ply_points, vis_points, model.images);
            std::string sample_path = JoinPaths(cluster_rect_path, "sampled_fused.ply");
            std::string sample_vis_path = JoinPaths(cluster_rect_path, "sampled_fused.ply") + ".vis";
            WriteBinaryPlyPoints(sample_path, ply_points, false, true);
            WritePointsVisibility(sample_vis_path, vis_points);
            return;
        }

        int temp_num_points = options_.sampInsert? 
            1e7 * options_.dist_insert * options_.max_ram / 50.0f : 5e7 * options_.max_ram / 200.0f;
        const int max_num_points = std::min(temp_num_points, (int)8e7);
        std::cout << "max_ram: " << options_.max_ram 
                  << "  inster_factor: " << options_.dist_insert
                  << "  max_num_points: " << max_num_points << std::endl;

        std::vector<std::vector<uint32_t> > cluster_point_ids;
        std::vector<std::vector<PlyPoint>> cluster_ply_points;
        std::vector<std::vector<std::vector<uint32_t>>> cluster_vis_points;
        std::vector<std::vector<std::vector<float>>> cluster_weight_points;
        std::vector<std::vector<float>> cluster_score_points;
        std::vector<struct MeshBox> cluster_mesh_box;
        int num_cluster = -1;
        if (ply_points.size() > max_num_points && options_.mesh_cluster /*&& lidar_points.empty()*/){
            std::vector<std::vector<std::size_t> > point_cluster_map;
            std::vector<struct Box> cluster_bound_box;

            num_cluster = PointsCluster(point_cluster_map, cluster_bound_box, 
                                        ply_points, model, max_num_points, 
                                        options_.overlap_factor, roi_meshbox);

            cluster_point_ids.resize(num_cluster);
            cluster_ply_points.resize(num_cluster);
            cluster_vis_points.resize(num_cluster);
            cluster_weight_points.resize(num_cluster);
            if (has_score){
                cluster_score_points.resize(num_cluster);
            }
            cluster_mesh_box.resize(num_cluster);
            int dou_points_num = 0;
            for (size_t i = 0; i < ply_points.size(); ++i) {
                for (auto id : point_cluster_map[i]){
                    cluster_point_ids.at(id).push_back(i);
                    dou_points_num++;
                }
            }
            for (int i = 0; i < num_cluster; i++){
                cluster_mesh_box.at(i) = 
                    cluster_bound_box.at(i).ToMeshBox();
            }
            // for (int i = 0; i < num_cluster; i++){
            //     std::string path = dense_reconstruction_path + "/" + to_string(i) + ".ply";
            //     WriteBinaryPlyPoints(path, cluster_ply_points[i]);
            // }
            std::cout << "repeat points num / ply points num: " 
                      << dou_points_num - ply_points.size() 
                      << " / " << ply_points.size() << std::endl;
        } else {
            num_cluster = 1;
            cluster_point_ids.resize(num_cluster);
            cluster_point_ids[0].resize(ply_points.size());
            std::iota(cluster_point_ids[0].begin(), cluster_point_ids[0].end(), 0);
            cluster_ply_points.resize(num_cluster);
            cluster_vis_points.resize(num_cluster);
            cluster_weight_points.resize(num_cluster);
            if (has_score){
                cluster_score_points.resize(num_cluster);
            }
        }

        for (int cluster_id = 0; cluster_id < num_cluster; cluster_id++){
            mvs::PointCloud pointcloud;
            // size_t num_vpoint = ply_points.size();
            size_t num_point = cluster_point_ids.at(cluster_id).size();
            // size_t num_point = num_vpoint + lidar_points.size();
            std::cout << "Cluster " << cluster_id 
                      << " points.size() = " << num_point << std::endl;

            if (num_point < 100 && num_cluster > 1){
                continue;
            }
            {
                pointcloud.points.Resize(num_point);
                pointcloud.colors.Resize(num_point);
                pointcloud.pointTypes.Resize(num_point);
                if (has_score){
                    pointcloud.scores.Resize(num_point);
                }
                pointcloud.pointViews.Resize(num_point);
                pointcloud.pointWeights.Resize(num_point);

                uint64_t indices_print_step = num_point / 10 + 1;
                #pragma omp parallel for
                for (size_t i = 0; i < num_point; ++i) {
                    if (i % indices_print_step == 0) {
                        std::cout << "\r" << i << " / " << num_point;
                    }
                    const int point_id = cluster_point_ids.at(cluster_id)[i];
                    const PlyPoint& pt = ply_points.at(point_id);
                    const std::vector<uint32_t>& vis_pt = vis_points.at(point_id);
                    const std::vector<float>& weight_pt = weight_points.at(point_id);
                    Eigen::Vector3f p(pt.x, pt.y, pt.z);
                    pointcloud.points.SetAt(i, p);
                    pointcloud.colors.SetAt(i, Eigen::Vector3ub(pt.r, pt.g, pt.b));
                    pointcloud.pointTypes.SetAt(i, pt.s_id);
                    if (has_score){
                        pointcloud.scores.SetAt(i, score_points.at(point_id));
                    }
                    for (int j = 0; j < vis_pt.size(); j++) {
                        const auto& view = vis_pt[j];
                        pointcloud.pointViews[i].Insert(view);
                        const auto& weight = weight_pt[j];
                        pointcloud.pointWeights[i].Insert(weight);
                    }
                }

                std::cout << std::endl;
            }

            auto images = model.images;
            std::cout << "image size, lidar size: " << images.size() << ", " << lidars_pose.size() << std::endl;
            if (!lidars_pose.empty()){
                images.insert(images.end(), lidars_pose.begin(), lidars_pose.end());
            }

            Timer dela_timer;
            dela_timer.Start();

            TriangleMesh obj_mesh_cluster;
            bool delaunay_flag = mvs::DelaunayMeshing(obj_mesh_cluster, pointcloud, images, 
                                                    options_.sampInsert, options_.dist_insert, options_.diff_depth, options_.plane_insert_factor, 
                                                    options_.plane_score_thred, false, 4, options_.sigma, 1, 4, 3, 0.1, 1000, 400);
            if (!delaunay_flag){
                std::cout << "Skip Cluster: " << cluster_id << std::endl;
                continue;
            }

            std::cout << "Meshing Delaunay Elapsed time:" << dela_timer.ElapsedMinutes() << "[minutes]\n" << std::endl;


#ifdef DELAUNAY_SAVE_POINTS
            int num_delaunay_points = pointcloud.points.size();
            std::vector<PlyPoint> delaunay_points;
            std::vector<PlyPoint> delaunay_vis_num_points;
            delaunay_points.reserve(num_delaunay_points);
            delaunay_vis_num_points.reserve(num_delaunay_points);

            std::vector<int> num_vis;
            int max_num_vis = 0;
            float max_weight = 0;
            float min_weight = 1000;
            float sum_all_weight = 0;
            std::vector<float> sum_weights;
            for (int i = 0; i < num_delaunay_points; i++) {
                const auto& pnt_weights = pointcloud.pointWeights[i];
                int num_views = pnt_weights.size();
                num_vis.push_back((int)num_views);
                float sum_weight = 0;
                for (int j = 0; j < num_views; j++){
                    sum_weight += pnt_weights[j];
                }
                if (sum_weight < min_weight){
                    min_weight = sum_weight;
                }
                if (sum_weight > max_weight){
                    max_weight = sum_weight;
                }
                sum_all_weight += sum_weight;
                sum_weights.push_back(sum_weight);
            }
            std::sort(num_vis.begin(), num_vis.end());
            int nth = num_vis.at(num_vis.size() * 0.95 - 1);
            std::cout << "\nDELAUNAY_SAVE_POINTS\nmax_num_vis: " << nth <<  std::endl;

            std::sort(sum_weights.begin(), sum_weights.end());
            float nth2 = sum_weights.at(sum_weights.size() * 0.08 + 1);
            float nth3 = sum_weights.at(sum_weights.size() * 0.8 - 1);
            std::cout << "\nDELAUNAY_SAVE_POINTS\nmax_num_weight: " << nth2 
                << "max_weight / min_weight: " << max_weight 
                << " / " << min_weight <<  std::endl;

            for (int i = 0; i < num_delaunay_points; i++){
                PlyPoint pnt;
                pnt.x = pointcloud.points[i].x();
                pnt.y = pointcloud.points[i].y();
                pnt.z = pointcloud.points[i].z();
                pnt.r = pointcloud.colors[i].x();
                pnt.g = pointcloud.colors[i].y();
                pnt.b = pointcloud.colors[i].z();

                // const auto& pnt_weights = pointcloud.pointWeights[i];
                // int num_views = pnt_weights.size();
                // float sum_weight = 0;
                // for (int j = 0; j < num_views; j++){
                //     sum_weight += pnt_weights[j];
                // }
                // if (sum_weight < nth2 + 5){
                //     continue;
                // }

                delaunay_points.push_back(pnt);

                const float gray = (float)std::min((int)pointcloud.pointViews[i].GetSize(), nth) / (float)nth;
                // const float gray = (float)std::min((float)sum_weight, nth3) / (float)nth3;
                ColorMap(gray, pnt.r, pnt.g, pnt.b);
                delaunay_vis_num_points.push_back(pnt);
            }
            std::cout << "Delaunay Size: " << delaunay_points.size() << std::endl;
            std::string delaunay_path = JoinPaths(cluster_rect_path, "delaunay_fused.ply");
            WriteBinaryPlyPoints(delaunay_path, delaunay_points, false, true);

            std::string delaunay_vis_path = JoinPaths(cluster_rect_path, "visNum_fused.ply");
            WriteBinaryPlyPoints(delaunay_vis_path, delaunay_vis_num_points, false, true);
            // return;
#endif

            // ColorizingMesh(obj_mesh_cluster, model.images);

            Timer adj_timer;
            adj_timer.Start();

            float fDecimate = 1.f;
            AdjustMesh(obj_mesh_cluster, fDecimate);

            std::cout << "Meshing AdjustMesh Elapsed time:" << adj_timer.ElapsedMinutes() << "[minutes]" << std::endl;

            if (cluster_mesh_box.size() > 1){
                cluster_mesh_box.at(cluster_id).ResetBoundary(0.5);
                FilterWithBox(obj_mesh_cluster, cluster_mesh_box.at(cluster_id));
            }

            std::cout << "Meshing Filter Elapsed time:" << adj_timer.ElapsedMinutes() << "[minutes]" << std::endl;

            fDecimate = 0.5;
            std::cout << "fDecimate: " << fDecimate << std::endl;
            obj_mesh_cluster.Clean(fDecimate, 0, false, 0, 0, false);
            obj_mesh_cluster.ComputeNormals();

            if (obj_mesh_cluster.vertices_.empty()|| obj_mesh_cluster.faces_.empty()){
                std::cout << "obj_mesh_cluster is empty" << std::endl;
                continue;
            }

            // obj_mesh.AddMesh(obj_mesh_cluster);
            if (num_cluster > 1){
                write_as_sem = write_as_sem && (!obj_mesh_cluster.vertex_labels_.empty());
                write_as_rgb = write_as_rgb && (!obj_mesh_cluster.vertex_colors_.empty());
                std::string path = JoinPaths(
                    output_path, std::to_string(cluster_id) + "-" + MODEL_NAME);
                if (write_as_sem && obj_mesh_cluster.vertex_labels_.size() != obj_mesh_cluster.vertices_.size()){
                    std::cout << "Info WriteTriangleMeshObj: " << write_as_sem 
                        << "; size: " << obj_mesh_cluster.vertices_.size() << " / "
                        << obj_mesh_cluster.vertex_colors_.size() << " / "
                        << obj_mesh_cluster.vertex_labels_.size() << std::endl;
                    obj_mesh_cluster.vertex_labels_.resize(obj_mesh_cluster.vertices_.size());
                }
                WriteTriangleMeshObj(path, obj_mesh_cluster, write_as_rgb, write_as_sem);
            }
            saved_cluster_ids.push_back(cluster_id);
            obj_mesh.Swap(obj_mesh_cluster);

            malloc_trim(0);

        }
    } else if (image_type_.compare("panorama") == 0) {
        std::vector<std::string> perspective_image_names;
        std::vector<mvs::Image> perspective_images;
        std::vector<image_t> image_ids;
        std::vector<std::vector<int> > overlapping_images;
        std::vector<std::pair<float, float> > depth_ranges;
        if (!ImportPanoramaWorkspace(dense_reconstruction_path, 
            perspective_image_names, perspective_images, image_ids,
            overlapping_images, depth_ranges, false)) {
            return ;
        }

        mvs::PointCloud pointcloud;
        {
            pointcloud.pointViews.Resize(ply_points.size());
            pointcloud.pointWeights.Resize(ply_points.size());
            for (size_t i = 0; i < ply_points.size(); ++i) {
                const PlyPoint& pt = ply_points[i];
                const std::vector<uint32_t>& vis_pt = vis_points[i];
                Eigen::Vector3f p(pt.x, pt.y, pt.z);
                pointcloud.points.Insert(p);
                Eigen::Vector3f pn(pt.nx, pt.ny, pt.nz);
                pointcloud.colors.Insert(Eigen::Vector3ub(pt.r, pt.g, pt.b));
                pointcloud.pointTypes.Insert(pt.s_id);
                for (const auto& view : vis_pt) {                 
                    pointcloud.pointViews[i].Insert(view);
                    pointcloud.pointWeights[i].Insert(1.0);
                }
            }
            vis_points.clear();
        }

        mvs::DelaunayMeshing(obj_mesh, pointcloud, perspective_images, options_.sampInsert, 
                             options_.dist_insert, options_.diff_depth, options_.plane_insert_factor, 
                             options_.plane_score_thred, false, 4, options_.sigma, 1, 4, 3, 0.1, 1000, 400);
        // ColorizingMesh(obj_mesh, perspective_images);
        float fDecimate = 1.0f;
        AdjustMesh(obj_mesh, fDecimate);
        saved_cluster_ids.push_back(0);
    }

    {
        std::vector<PlyPoint>().swap(ply_points);
        std::vector<std::vector<uint32_t>>().swap(vis_points);
        std::vector<std::vector<float>>().swap(weight_points);
        if (has_score){
            std::vector<float>().swap(score_points);
        }
    }

    if (saved_cluster_ids.size() > 1){
        std::vector<TriangleMesh> cluster_obj_mesh(saved_cluster_ids.size() - 1);
        auto ReadMesh = [&](const int cluster_id){
            std::string path = JoinPaths(
                output_path, std::to_string(cluster_id) + "-" + MODEL_NAME);
            ReadTriangleMeshObj(path, cluster_obj_mesh.at(cluster_id), 
                                write_as_rgb, write_as_sem);
        };

        int num_eff_threads = GetEffectiveNumThreads(-1);
        num_eff_threads = std::min(num_eff_threads, (int)cluster_obj_mesh.size());
        std::unique_ptr<ThreadPool> thread_pool;
        thread_pool.reset(new ThreadPool(num_eff_threads));
        std::cout << "Load mesh num_eff_threads: " << num_eff_threads << std::endl;
        for (int i = 0; i < cluster_obj_mesh.size(); i++){
            int idx = saved_cluster_ids.at(i);
            thread_pool->AddTask(ReadMesh, idx);
        }
        thread_pool->Wait();

        for (int i = 0; i < cluster_obj_mesh.size(); i++){
            obj_mesh.AddMesh(cluster_obj_mesh.at(i));
            TriangleMesh temp_mesh;
            cluster_obj_mesh.at(i).Swap(temp_mesh);
        }
        cluster_obj_mesh.clear();
        cluster_obj_mesh.shrink_to_fit();
        std::cout << "obj_mesh info (vertices size, color size, normals size, labels size): " 
                << obj_mesh.vertices_.size() << ", " 
                << obj_mesh.vertex_colors_.size() << ", "
                << obj_mesh.vertex_normals_.size() << ", "
                << obj_mesh.vertex_labels_.size() << std::endl;
    }
#else
    ReadTriangleMeshObj(output_obj_path + "-orig.obj", obj_mesh, 
                                write_as_rgb, write_as_sem);
    AdjustMesh(obj_mesh);
    WriteTriangleMeshObj(output_obj_path + "-adjust1.obj", obj_mesh);
# endif

    if (options_.roi_mesh && ExistsFile(ori_box_path)){
        // WriteTriangleMeshObj(output_obj_path + "-ori.obj", obj_mesh);
        roi_meshbox.ResetBoundary(0.5);
        FilterWithBox(obj_mesh, roi_meshbox);
    }
    if (!(obj_mesh.vertices_.empty()|| obj_mesh.faces_.empty())){

        // double fDecimate = 0.5;
        // std::cout << "fDecimate: " << fDecimate << std::endl;
        // obj_mesh.Clean(fDecimate, 0, false, 0, 0, false);
        // obj_mesh.ComputeNormals();
        WriteTriangleMeshObj(output_obj_path, obj_mesh);

        bool has_semantic = false;
        for (size_t i = 0; i < obj_mesh.vertices_.size(); ++i) {
            int8_t label = obj_mesh.vertex_labels_.at(i);
            if (label != -1) {
                has_semantic = true;
                break;
            }
        }
        
        if (has_semantic) {
            TriangleMesh sem_mesh = obj_mesh;
            RefineSemantization(sem_mesh);
            Fix(sem_mesh);
            for (size_t i = 0; i < sem_mesh.vertices_.size(); ++i) {
                int best_label = sem_mesh.vertex_labels_[i];
                best_label = (best_label + 256) % 256;
                Eigen::Vector3d rgb;
                rgb[0] = adepallete[best_label * 3];
                rgb[1] = adepallete[best_label * 3 + 1];
                rgb[2] = adepallete[best_label * 3 + 2];
                sem_mesh.vertex_colors_[i] = rgb;
            }
            WriteTriangleMeshObj(output_objsem_path, sem_mesh, true, true);
        }
    }
}

void DelaMeshing::AdjustMesh(TriangleMesh& obj_mesh, float fDecimate = 1.0f){

    Timer modi_timer;
    modi_timer.Start();
    obj_mesh.ModifyNonMainfoldFace();
    
    std::cout << "AdjustMesh ModifyNonMainfoldFace Elapsed time:" << modi_timer.ElapsedMinutes() << "[minutes]" << std::endl;

    // clean the mesh
    
    modi_timer.Restart();
    obj_mesh.Clean(options_.decimate_mesh * fDecimate, 
                  options_.only_remove_edge_spurious ? -1.0f : options_.remove_spurious,
                  options_.remove_spikes, 
                  options_.close_holes, 
                  options_.smooth_mesh * fDecimate, true);
    
    if (options_.only_remove_edge_spurious && options_.remove_spurious > 1.0f){
        obj_mesh.FilterOutOfRangeFace(options_.remove_spurious);
    }

    // obj_mesh.Clean(1.f, 0.f, options_.remove_spikes, 
    //                 options_.close_holes, 0, false);
    
    // extra cleaning to remove non-manifold problems created by closing 
    // holes.
    // obj_mesh.Clean(1.f, 0.f, false, 0, 0, true);

    modi_timer.Restart();
    obj_mesh.RemoveIsolatedPieces(options_.num_isolated_pieces * fDecimate);
    std::cout << "AdjustMesh RemoveIsolatedPieces Elapsed time:" << modi_timer.ElapsedMinutes() << "[minutes]" << std::endl;

    // Remove abnormal facets & Hole filling.
    if(options_.fix_mesh) {
        std::vector<int> border_verts_idx;
        obj_mesh.RemoveAbnormalFacets(border_verts_idx);

        MeshInfo info;
        prepare_mesh(&info, obj_mesh);

        //remove undesired faces
        info.remove_complex_faces(obj_mesh);
        info.remove_adj_border_faces(obj_mesh);
        info.remove_complex_faces(obj_mesh);

        //get border list by border vertices
        std::vector<std::vector<int> > lists;
        info.get_border_lists(border_verts_idx, lists);

        for(int i = lists.size() - 1; i >= 0; --i){
            if(lists[i].size() > options_.close_holes)
                lists.erase(lists.begin() + i);
        }

        //hole filling by given border list
        obj_mesh.HollFill(lists);

        //remove isolated vertices
        info.clear();
        info.initialize(obj_mesh);
        info.remove_unref_vertices(obj_mesh);
    }
    std::cout << "AdjustMesh fix_mesh Elapsed time:" << modi_timer.ElapsedMinutes() << "[minutes]" << std::endl;

    //recompute normals
    obj_mesh.ComputeNormals();
}

void DelaMeshing::ColorizingMesh(TriangleMesh& obj_mesh, const std::vector<mvs::Image>& images) {
    Timer timer;
    timer.Start();
    std::cout << "Colorizing Mesh" << std::endl;
    if (obj_mesh.vertices_.size() != obj_mesh.vertex_visibilities_.size()) {
        std::cout << "failed to colorizing!" << std::endl;
        return ;
    }
    std::vector<std::vector<size_t> > image_points_indices;
    image_points_indices.resize(images.size());
    for (size_t i = 0; i < obj_mesh.vertex_visibilities_.size(); ++i) {
        for (auto & vis : obj_mesh.vertex_visibilities_.at(i)) {
            image_points_indices[vis].push_back(i);
        }
    }

    size_t progress = 0;

    std::vector<int> hits(obj_mesh.vertices_.size(), 0);
    std::vector<Eigen::Vector3d> colors(obj_mesh.vertices_.size());
    std::fill(colors.begin(), colors.end(), Eigen::Vector3d::Zero());
    for (size_t i = 0; i < images.size(); ++i) {
        if (image_points_indices.at(i).empty()) {
            continue;
        }
        const mvs::Image & image = images.at(i);
        const int width = image.GetWidth();
        const int height = image.GetHeight();
        Eigen::RowMatrix3x4d P = Eigen::RowMatrix3x4f(image.GetP()).cast<double>();

        Bitmap bitmap;
        if (!bitmap.Read(image.GetPath())) {
            std::cout << "failed to load image " << image.GetPath() << std::endl;
            continue;
        }
        for (auto point_idx : image_points_indices.at(i)) {
            Eigen::Vector4d hX = obj_mesh.vertices_.at(point_idx).homogeneous();
            Eigen::Vector3d proj = P * hX;
            int u = proj.x() / proj.z();
            int v = proj.y() / proj.z();
            if (u < 0 || u >= width || v < 0 || v >= height) {
                continue;
            }
            BitmapColor<uint8_t> color = bitmap.GetPixel(u, v);
            colors[point_idx].x() += color.r;
            colors[point_idx].y() += color.g;
            colors[point_idx].z() += color.b;
            hits[point_idx]++;
        }
        bitmap.Deallocate();
        if ((++progress) % 100 == 0) {
            std::cout << StringPrintf("\rProcess Image#%d", progress);
        }
    }
    std::cout << std::endl;
    if (obj_mesh.vertex_colors_.empty()) {
        obj_mesh.vertex_colors_.resize(obj_mesh.vertices_.size());
    }
    for (size_t i = 0; i < obj_mesh.vertices_.size(); ++i) {
        if (hits[i]) {
            obj_mesh.vertex_colors_.at(i) = colors[i] / hits[i];
        }
    }
    timer.PrintMinutes();
}

void DelaMeshing::MergeMeshing(const std::string &workspace_path){
    const auto& dense_reconstruction_path = JoinPaths(workspace_path, DENSE_DIR);
    if (ExistsFile(JoinPaths(dense_reconstruction_path, MODEL_NAME))){
        return;
    //   boost::filesystem::remove(JoinPaths(dense_reconstruction_path, MODEL_NAME));
    //   std::cout << "remove file: " << JoinPaths(dense_reconstruction_path, MODEL_NAME) << std::endl;
    }
    // if (ExistsFile(JoinPaths(dense_reconstruction_path, SEM_MODEL_NAME))){
    //   boost::filesystem::remove(JoinPaths(dense_reconstruction_path, SEM_MODEL_NAME));
    //   std::cout << "remove file: " << JoinPaths(dense_reconstruction_path, SEM_MODEL_NAME) << std::endl;
    // }

    TriangleMesh merge_mesh;
    TriangleMesh merge_sem_mesh;

    for (int rect_id = 0; ; rect_id++){
        auto reconstruction_path =
            JoinPaths(workspace_path, std::to_string(rect_id));
        if (!ExistsDir(reconstruction_path)){
            break;
        }
        std::cout << "Merge Reconstruction: " << rect_id << std::endl;


        std::string mesh_input_path = JoinPaths(reconstruction_path.c_str(), MODEL_NAME);
        if (ExistsFile(mesh_input_path)) {
            TriangleMesh mesh;
            ReadTriangleMeshObj(mesh_input_path, mesh, true);
            merge_mesh.AddMesh(mesh);
            std::cout << "=> merge :" << mesh_input_path << std::endl;
        }

        std::string mesh_sem_input_path = JoinPaths(reconstruction_path.c_str(), SEM_MODEL_NAME);
        if (ExistsFile(mesh_sem_input_path)) {
            TriangleMesh sem_mesh;
            ReadTriangleMeshObj(mesh_sem_input_path, sem_mesh, true, true);
            merge_sem_mesh.AddMesh(sem_mesh);
            std::cout << "=> merge :" << mesh_sem_input_path << std::endl;
        }
    }


    if (merge_mesh.vertices_.size() > 0 && merge_mesh.faces_.size() > 0){
        std::string output_path = JoinPaths(dense_reconstruction_path, MODEL_NAME);
        WriteTriangleMeshObj(output_path, merge_mesh, true);
        std::cout << "=> save " << MODEL_NAME << std::endl;
    }

    if (merge_sem_mesh.vertices_.size() > 0 && merge_sem_mesh.faces_.size() > 0){
        std::string output_path = JoinPaths(dense_reconstruction_path, SEM_MODEL_NAME);
        WriteTriangleMeshObj(output_path, merge_sem_mesh, true, true);
        std::cout << "=> save " << SEM_MODEL_NAME << std::endl;
    }

    std::cout << "\nDone, save merge result to " << dense_reconstruction_path << std::endl;
}

}  // namespace mvs
}  // namespace sensemap
