// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include "mvs/reconstruction_cluster.h"
#include "util/common.h"

#include "yaml-cpp/yaml.h"

namespace sensemap {
namespace mvs {

void GreyToColorMix(int id, Eigen::Vector3ub &rgb)
{
    cv::RNG rng(uint64(id * 1.0e2));
    rgb.x() = rng.uniform(0, 255);
    rgb.y() = rng.uniform(0, 255);
    rgb.z() = rng.uniform(0, 255);
};

void WritePLY(std::shared_ptr<Reconstruction> reconstruction, const std::string &path) {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    // Calculate pose number and mappoint number
    const int mappoint_num = reconstruction->NumMapPoints();

    file << "ply" << std::endl;
    file << "format ascii 1.0" << std::endl;
    file << "element vertex " << mappoint_num << std::endl;
    file << "property float x" << std::endl;
    file << "property float y" << std::endl;
    file << "property float z" << std::endl;
    file << "property uchar red" << std::endl;
    file << "property uchar green" << std::endl;
    file << "property uchar blue" << std::endl;
    file << "end_header" << std::endl;

    // Output mappoints
    for (const auto mappoint_id : reconstruction->MapPointIds()) {
        auto cur_mappoint = reconstruction->MapPoint(mappoint_id);

        std::ostringstream line;
        line << cur_mappoint.X() << " ";
        line << cur_mappoint.Y() << " ";
        line << cur_mappoint.Z() << " ";
        line << static_cast<int>(cur_mappoint.Color(0)) << " ";
        line << static_cast<int>(cur_mappoint.Color(1)) << " ";
        line << static_cast<int>(cur_mappoint.Color(2));
        file << line.str() << std::endl;
    }

    file.close();
}

bool ReconstructionCluster::Options::SetCellSize(const std::vector<Eigen::Vector3f>& poses){
    const int num_images = poses.size();
    float grid_size = cluster_num;

    // Transform points & Calculate BoundingBox.
    Eigen::Vector3f box_min, box_max;
    box_min = box_max = poses[0];
    for (int i = 0; i < num_images; ++i) {
        auto& point = poses[i];
        
        box_min[0] = std::min(box_min[0], point[0]);
        box_min[1] = std::min(box_min[1], point[1]);
        box_min[2] = std::min(box_min[2], point[2]);
        box_max[0] = std::max(box_max[0], point[0]);
        box_max[1] = std::max(box_max[1], point[1]);
        box_max[2] = std::max(box_max[2], point[2]);
    }

    max_cell_size = std::sqrt((box_max.x() - box_min.x()) * 
                (box_max.y() - box_min.y()) / grid_size);
    
    if (max_cell_size > (box_max.x() - box_min.x())){
        max_cell_size = std::max(max_cell_size,
                        (box_max.y() - box_min.y()) / grid_size);
    } else if(max_cell_size > (box_max.y() - box_min.y())) {
        max_cell_size = std::max(max_cell_size,
                        (box_max.x() - box_min.x()) / grid_size);
    }

    if (grid_size <= 1){
        max_cell_size = 
            std::max(box_max.y() - box_min.y(), box_max.x() - box_min.x());
    }

    return true;
}

void ReconstructionCluster::Options::Print() const {
    PrintHeading2("Reconstruction Cluster::Options");
    PrintOption(cluster_num);
    PrintOption(max_ram);
    PrintOption(ram_eff_factor);
    PrintOption(max_num_images);
    PrintOption(max_num_images_factor);
    PrintOption(min_pts_per_cluster);
    PrintOption(valid_spacing_factor);
    PrintOption(min_common_view);
    PrintOption(max_filter_percent);
    PrintOption(dist_threshold);
    PrintOption(max_cell_size);
    PrintOption(min_cell_size);
    PrintOption(outlier_spacing_factor);
    std::cout << std::endl;
}

bool ReconstructionCluster::Options::Check() const {
    CHECK_OPTION_GT(min_pts_per_cluster, 0);
    CHECK_OPTION_GT(max_cell_size, 0.0f);
    CHECK_OPTION_GE(valid_spacing_factor, 0.0);
    return true;
}

ReconstructionCluster::ReconstructionCluster(
    Options& options,
    std::string out_workspace_path)
    :options_(options),
     num_out_reconstructions_(0),
     out_workspace_path_(out_workspace_path){};

void ReconstructionCluster::Run(){

    int num_images_clustered = 0;    
    int num_images_ori = 0;
    
    CHECK(boost::filesystem::exists(out_workspace_path_));

    std::vector<std::shared_ptr<Reconstruction> > reconstructions;
    size_t reconstruction_idx = 0;
    reconstructions.clear();
    for (size_t rec_idx = 0; ; rec_idx++) {
        auto reconstruction_path = JoinPaths(out_workspace_path_, 
            std::to_string(reconstruction_idx), DENSE_DIR, SPARSE_DIR);
        if (!ExistsDir(reconstruction_path)) {
            break;
        }
        
        PrintHeading1(StringPrintf("Reconstruction# %d", rec_idx));
        reconstruction_.reset(new Reconstruction());
        reconstruction_->ReadReconstruction(reconstruction_path);
        num_images_ori = reconstruction_->NumRegisterImages();

        int init_num = InitParam(rec_idx);
       
#ifdef OLD_FILE_SYATEM
        if (init_num == 1){
            std::cout << "Number of Tile Cluster / Ori："  
                << 1 << " / " <<  1 << std::endl;
            continue;
        }
        if (rec_idx == 0){
            InitWrokspace();
        }
#else
        out_rect_path_ = JoinPaths(out_workspace_path_, std::to_string(rec_idx));
#endif

        cluster_num_ = ClusterRun();
        num_images_clustered += SaveClusteredRect();
        num_out_reconstructions_ += cluster_num_;

        reconstruction_idx++;
    }
    
    std::cout << "Number of Tile Cluster / Ori：" 
              << cluster_num_ << " / " << reconstruction_idx
              << "\nRepeat: " << (float)1.0f 
              << "\tCluster: " << cluster_num_ << std::endl;
}

void ReconstructionCluster::InitWrokspace(){
    std::string back = out_workspace_path_.substr(out_workspace_path_.length() - 1 , 1);
    if (back.compare("/") == 0){
        ori_workspace_path_ = out_workspace_path_.substr(0, out_workspace_path_.length() - 1) + WORKSPACE_ORI;
    } else {
        ori_workspace_path_ = out_workspace_path_ + WORKSPACE_ORI;
    }
    
    CHECK(!boost::filesystem::exists(ori_workspace_path_));
    CHECK(boost::filesystem::exists(out_workspace_path_));

    boost::filesystem::rename(out_workspace_path_, ori_workspace_path_);

    if (!boost::filesystem::exists(out_workspace_path_)) {
        CreateDirIfNotExists(out_workspace_path_);
    }
    
    std::vector<std::string> file_names = GetFileList(ori_workspace_path_);
    for(int iter = 0; iter < file_names.size(); iter++){
        std::string file_name = GetPathBaseName(file_names.at(iter));
        boost::filesystem::copy_file(JoinPaths(ori_workspace_path_, file_name),
                                    JoinPaths(out_workspace_path_, file_name));
        // std::cout << file_name << std::endl;
    }
}

void ReconstructionCluster::ReadPriorBox(size_t rec_idx){
        std::string ori_box_path = JoinPaths(out_workspace_path_, 
            std::to_string(rec_idx), ROI_BOX_NAME);
        std::cout << "ori_box_path: " << ori_box_path << std::endl;
        if (ExistsFile(ori_box_path)){
            std::cout << "load box file(" << ori_box_path << ")" << std::endl;
            ReadBoundBoxText(ori_box_path, prior_box_);
            prior_box_.SetBoundary(1e-6, -1);
            prior_box_.z_box_min = -FLT_MAX;
            prior_box_.z_box_max = FLT_MAX;
            has_box_prior_ = true;

            std::cout << StringPrintf("ROI(box.txt): [%f %f %f] -> [%f %f %f]\n", 
                    prior_box_.x_box_min, prior_box_.y_box_min, prior_box_.z_box_min,
                    prior_box_.x_box_max, prior_box_.y_box_max, prior_box_.z_box_max);
        } else {
            std::cout << "!ExistsFile(" << ori_box_path << ")" << std::endl;
        } 
        return;
}


void ReconstructionCluster::ComputePriorBox(size_t rec_idx){
        std::cout << "init prior box " << std::endl;
        std::vector<Eigen::Vector3f> all_pnts = points_;
        all_pnts.insert(all_pnts.end(), poses_.begin(), poses_.end());

        prior_box_.x_min = prior_box_.y_min = prior_box_.z_min = FLT_MAX;
        prior_box_.x_max = prior_box_.y_max = prior_box_.z_max = -FLT_MAX;
        for (auto point : all_pnts) {
            prior_box_.x_min = std::min(prior_box_.x_min, point.x());
            prior_box_.y_min = std::min(prior_box_.y_min, point.y());
            prior_box_.z_min = std::min(prior_box_.z_min, point.z());
            prior_box_.x_max = std::max(prior_box_.x_max, point.x());
            prior_box_.y_max = std::max(prior_box_.y_max, point.y());
            prior_box_.z_max = std::max(prior_box_.z_max, point.z());
        }
        float x_offset = (prior_box_.x_max - prior_box_.x_min) * 0.05;
        float y_offset = (prior_box_.y_max - prior_box_.y_min) * 0.05;
        float z_offset = (prior_box_.z_max - prior_box_.z_min) * 0.05;
        prior_box_.x_box_min = prior_box_.x_min - x_offset;
        prior_box_.x_box_max = prior_box_.x_max + x_offset;
        prior_box_.y_box_min = prior_box_.y_min - y_offset;
        prior_box_.y_box_max = prior_box_.y_max + y_offset;
        prior_box_.z_box_min = prior_box_.z_min - z_offset;
        prior_box_.z_box_max = prior_box_.z_max + z_offset;
        prior_box_.rot = pivot_;
        has_box_prior_ = false;
        
        std::cout << StringPrintf("ROI(compute): [%f %f %f] -> [%f %f %f]\n", 
                prior_box_.x_box_min, prior_box_.y_box_min, prior_box_.z_box_min,
                prior_box_.x_box_max, prior_box_.y_box_max, prior_box_.z_box_max);
        return;
}

int ReconstructionCluster::InitParam(int rect_id = 0){
    num_points_ = reconstruction_->NumMapPoints();
    points_.resize(num_points_);
    point_idx_.resize(num_points_);
    point_visibility_.resize(num_points_);

    std::size_t num_images = reconstruction_->NumRegisterImages();
    poses_.resize(num_images);

    std::vector<Eigen::Vector3f> temp_points(num_points_);
    double sum_pt_dist = 0;
    uint64_t num_pt_dist = 0;
    auto map_points = reconstruction_->MapPoints();
    int i = 0;
    for (auto map_point : map_points){
        Eigen::Vector3f point;
        point.x() = (float)map_point.second.X();
        point.y() = (float)map_point.second.Y();
        point.z() = (float)map_point.second.Z();
        temp_points[i] = point;

        point_idx_[i] = map_point.first;

        const Track track = map_point.second.Track();
        for (const auto ele : track.Elements()){
            point_visibility_[i].emplace(ele.image_id);

            const auto& image = reconstruction_->Image(ele.image_id);
            image.ProjectionCenter();
            Eigen::Vector3f camera_c((float)image.ProjectionCenter().x(), 
                                     (float)image.ProjectionCenter().y(),
                                     (float)image.ProjectionCenter().z());
            sum_pt_dist += (point - camera_c).norm();
            num_pt_dist++;
        }
        i++;
    }
    float dist_pnt2camera = sum_pt_dist / (float)num_pt_dist;
    std::cout << "Reconstrution has " << i << " points, mean dist " << sum_pt_dist / num_pt_dist << std::endl;
    
    ReadPriorBox(rect_id);

    if (has_box_prior_){
        pivot_ = prior_box_.rot;
    } else {
        pivot_ = ComputePovitMatrix(temp_points);
    }

    for (int i = 0; i < points_.size(); ++i) {
        points_[i] = pivot_ * temp_points[i];
    }

    int idx = 0;
    int image_max_width = 0, image_max_height = 0;
    float image_max_fov = -1.0;
    images_numpoints_.reserve(reconstruction_->NumRegisterImages());
    for (auto image : reconstruction_->Images()){
        if (!image.second.IsRegistered()){
            continue;
        }
        Eigen::Vector3f pose;
        pose = image.second.ProjectionCenter().cast<float>();
        poses_[idx] = pivot_ * pose;

        images_numpoints_.emplace(image.first, image.second.NumMapPoints());

        const auto& camera = reconstruction_->Camera(image.second.CameraId());
        if (image_max_width < camera.Width()){
            image_max_width = camera.Width();
        }
        if (image_max_height < camera.Height()){
            image_max_height = camera.Height();
        }
        float fov;
        if (camera.MeanFocalLength() > 1e-4){
            float fov_h = (float)camera.Height() / camera.MeanFocalLength();
            float fov_w = (float)camera.Height() / camera.MeanFocalLength();
            fov = std::max(fov_w, fov_h);
        } else {
            fov = 1.0;
        }
        if (image_max_fov < fov){
            image_max_fov = fov;
        }
        
        idx++;
    }

    if(!has_box_prior_){
        ComputePriorBox(rect_id);
    }

    if (options_.max_image_size > 0){
        if (image_max_width > image_max_height){
            image_max_height = image_max_height  * options_.max_image_size / (float)image_max_width;
            image_max_width = options_.max_image_size;
        } else {
            image_max_width = image_max_height  * options_.max_image_size / (float)image_max_height;
            image_max_height = options_.max_image_size;
        }
    }
    if (options_.max_ram > 0){
        float image_memory = image_max_width * image_max_height * (3 + 4 + 12 + 1 + 1);
        uint64_t G_byte =  1.0e9;
        options_.max_num_images = options_.ram_eff_factor * options_.max_ram * G_byte/ image_memory;
        options_.cluster_num = std::ceil(reconstruction_->Images().size() / 
            (options_.max_num_images * options_.max_num_images_factor));
    }

    if (options_.max_cell_size < 0){
        options_.SetCellSize(poses_);
        options_.min_pts_per_cluster = std::min(options_.min_pts_per_cluster, 
            int(num_points_ / (5*options_.cluster_num)));
    } else {
        options_.min_pts_per_cluster = 0;
    }

    if (options_.min_cell_size < 0){
        std::cout << "dist_pnt2camera, image_max_fov: " << dist_pnt2camera << ", " << image_max_fov << std::endl;
        float min_cell_size = std::max(dist_pnt2camera * image_max_fov * 1.98f, options_.max_cell_size * 0.24f);
        options_.min_cell_size = std::min(min_cell_size, options_.max_cell_size * 0.49f);
    }

    options_.Print();
    return options_.cluster_num;

    // std::cout << images_numpoints_.size() << std::endl;
}

std::size_t ReconstructionCluster::ClusterRun() {
    std::cout << "Points Cluster::Run" << std::endl;
    std::size_t point_num = points_.size();

    // Transform points & Calculate BoundingBox.
    Eigen::Vector3f box_min, box_max, points_box_min, points_box_max;
    box_min = box_max = poses_[0];
    std::size_t pose_num = poses_.size();
    for (int j = 0; j < pose_num; ++j) {
        // auto pose = pivot_ * poses_[j];
        auto pose = poses_[j];
        
        box_min[0] = std::min(box_min[0], pose[0]);
        box_min[1] = std::min(box_min[1], pose[1]);
        box_min[2] = std::min(box_min[2], pose[2]);
        box_max[0] = std::max(box_max[0], pose[0]);
        box_max[1] = std::max(box_max[1], pose[1]);
        box_max[2] = std::max(box_max[2], pose[2]);
    }

    std::vector<float> point_spacings;
    float average_spacing = ComputeAvergeSapcing(point_spacings, points_);

    points_box_min = points_box_max = points_[0];
    std::vector<Eigen::Vector3f> transformed_points(point_num);
    for (int i = 0; i < point_num; ++i) {
        auto &point = transformed_points[i];
        // point = pivot_ * points_[i];
        point = points_[i];
        if (options_.outlier_spacing_factor > 0 &&
            point_spacings[i] > options_.outlier_spacing_factor * average_spacing){
            continue;
        }

        if (point(0) < prior_box_.x_box_min || point(0) > prior_box_.x_box_max ||
            point(1) < prior_box_.y_box_min || point(1) > prior_box_.y_box_max ||
            point(2) < prior_box_.z_box_min || point(2) > prior_box_.z_box_max){
            continue;
        }

        points_box_min[0] = std::min(points_box_min[0], point[0]);
        points_box_min[1] = std::min(points_box_min[1], point[1]);
        points_box_min[2] = std::min(points_box_min[2], point[2]);
        points_box_max[0] = std::max(points_box_max[0], point[0]);
        points_box_max[1] = std::max(points_box_max[1], point[1]);
        points_box_max[2] = std::max(points_box_max[2], point[2]);
    }
    const float cell_size = options_.max_cell_size;
    const std::size_t grid_size_x = static_cast<std::size_t>(std::ceil((box_max.x() - box_min.x()) / cell_size));
    const std::size_t grid_size_y = static_cast<std::size_t>(std::ceil((box_max.y() - box_min.y()) / cell_size));
    const std::size_t grid_side = grid_size_x;
    std::size_t grid_slide = grid_side * grid_size_y;
    // std::cout << "\n => box-boundary:\t(" << box_min.x() << "," << box_min.y() 
    //           << ")-(" << box_max.x() << " " << box_max.y() << ")" << std::endl;

    double delt_x = (grid_size_x * cell_size - box_max.x() + box_min.x()) / 2;
    double delt_y = (grid_size_y * cell_size - box_max.y() + box_min.y()) / 2;
    box_min[0] -= delt_x;
    box_min[1] -= delt_y;
    box_max[0] += delt_x;
    box_max[1] += delt_y;

    std::cout << " => average_spacing: " << average_spacing
              << "\n => grid_size_x: " << grid_size_x
              << "\n => grid_size_y: " << grid_size_y
              << "\n => grid_slide: " << grid_slide
              << "\n => box-boundary:\t(" << box_min.x() << "," << box_min.y() 
              << ")-(" << box_max.x() << " " << box_max.y() << ")" 
              << "\n => point-boundary:\t(" << points_box_min.x() << "," 
              << points_box_min.y() << ")-("  << points_box_max.x() << " " 
              << points_box_max.y() << ")\n" << std::endl;

    std::vector<std::size_t> cell_point_count(grid_slide, 0);
    std::vector<std::unordered_map<image_t, ImageVisibility>> cell_image_visib(grid_slide);
    std::vector<std::size_t> point_cell_map(point_num);
    // std::vector<float> cell_average_spacing(grid_slide, 0.0f);

    std::unordered_set<int> oulier_ids;
    for (std::size_t i = 0; i < point_num; ++i) {
        const auto &point = points_[i];
        int x_cell_temp = static_cast<int>((point.x() - box_min.x()) / cell_size);
        int y_cell_temp = static_cast<int>((point.y() - box_min.y()) / cell_size);
        std::size_t x_cell = static_cast<std::size_t>(std::max(0, std::min(x_cell_temp, (int)grid_size_x-1)));
        std::size_t y_cell = static_cast<std::size_t>(std::max(0, std::min(y_cell_temp, (int)grid_size_y-1)));

        if (point.x() < points_box_min.x() || point.x() > points_box_max.x() ||
            point.y() < points_box_min.y() || point.y() > points_box_max.y()){
            oulier_ids.emplace(i);
        }

        std::size_t cell_idx = y_cell * grid_side + x_cell;
        cell_point_count[cell_idx]++;
        point_cell_map[i] = cell_idx;
        // cell_average_spacing[cell_idx] += point_spacings[i];

        for (auto image_id : point_visibility_[i]){
            if (cell_image_visib[cell_idx].find(image_id) 
                == cell_image_visib[cell_idx].end()){
                cell_image_visib[cell_idx].emplace(image_id, 
                ImageVisibility{image_id, 1, images_numpoints_[image_id]});
            } else{
                cell_image_visib[cell_idx][image_id].num_visib_mappoint++;
            }
        }
    }
    std::cout << "outlier : " << oulier_ids.size() << " / " << point_num << std::endl;
    
    // bound box
    std::vector<Box> cell_bound_boxs(grid_slide);
    for (std::size_t cell_idx = 0; cell_idx < grid_slide; cell_idx++){
        int y_cell = cell_idx / grid_size_x;
        int x_cell = cell_idx % grid_size_x;

        Box box;
        box.x_min = x_cell == 0 ? prior_box_.x_box_min : x_cell * cell_size + box_min.x();
        box.y_min = y_cell == 0 ? prior_box_.y_box_min: y_cell * cell_size + box_min.y();
        box.z_min = prior_box_.z_box_min;
        box.x_max = x_cell == (grid_size_x - 1) ? prior_box_.x_box_max : (x_cell + 1) * cell_size + box_min.x();
        box.y_max = y_cell == (grid_size_y - 1) ? prior_box_.y_box_max : (y_cell + 1) * cell_size + box_min.y();
        box.z_max = prior_box_.z_box_max;
        // std::cout << "id: " << cell_idx << "(" << box.x_min << " " << box.y_min << ")-(" << box.x_max << " " << box.y_max << ")" << std::endl;
        box.rot = pivot_;
        cell_bound_boxs[cell_idx] = box;
    }

    std::vector<int> cell_cluster_map;
    const float valid_spacing = average_spacing * options_.valid_spacing_factor;
#ifndef DynamicSize
    cluster_num_
        = GridCluster(cell_cluster_map, cell_point_count,cell_image_visib, 
                      cell_bound_boxs, point_cell_map);
#else
    cluster_num_
        = CommonViewCluster(cell_cluster_map, cell_image_visib, cell_bound_boxs, 
                            cell_point_count, grid_size_x, grid_size_y, valid_spacing);
#endif

    point_cluster_map_.resize(point_num);
    memset(point_cluster_map_.data(), -1, point_num * sizeof(int));
    for (std::size_t i = 0; i < point_num; ++i) {
        if (oulier_ids.find(i) != oulier_ids.end()){
            continue;
        }
        point_cluster_map_[i] = cell_cluster_map[point_cell_map[i]];
    }

    return cluster_num_;
}

int ReconstructionCluster::ImageVisibilityInsert(std::unordered_map<image_t, ImageVisibility>& imgvis_um1,
                          const std::unordered_map<image_t, ImageVisibility>& imgvis_um2){
    for (auto imgvis : imgvis_um2){
        if (imgvis_um1.find(imgvis.first) == imgvis_um1.end()){
            imgvis_um1.emplace(imgvis.first, imgvis.second);
        } else {
            imgvis_um1[imgvis.first].num_visib_mappoint 
                += imgvis.second.num_visib_mappoint;
        }
    }
    return imgvis_um1.size();
};

int ReconstructionCluster::ImageVisibilityFilter(const std::unordered_map<image_t, ImageVisibility>& imgvis_um,
                          const Options options){
    std::vector<std::pair<image_t, float>> vt_umap;
    for (auto it = imgvis_um.begin(); it != imgvis_um.end(); it++){
        if (it->second.num_all_mappoint == 0){
            continue;
        }
        vt_umap.push_back(std::make_pair(it->first, 
                (float)it->second.num_visib_mappoint / it->second.num_all_mappoint));
    }

    std::sort(vt_umap.begin(), vt_umap.end(), 
              [](std::pair<image_t, float> &a, std::pair<image_t, float> &b)
              {return a.second < b.second;});
    
    int num_filter_image = 0;
    for (int id = 0; id < vt_umap.size() * options.max_filter_percent; id++){
        if (vt_umap[id].second > options.min_common_view){
            break;
        }
        num_filter_image++;
    }
    // std::cout << imgvis_um.size() << " " << vt_umap.size() << " " << num_filter_image 
    //           << " " << options.max_filter_percent 
    //           << " " << options.min_common_view << std::endl;
    return num_filter_image;
}

bool ReconstructionCluster::WhetherMerge(const std::vector<Box> &cell_bound_box,
                  const int cell_idx, const int merge_cluster_idx){
    bool x_flag = 
        (std::abs(cell_bound_box[cell_idx].x_min - 
        cluster_bound_box_[merge_cluster_idx].x_min) < EPSILON) && 
        (std::abs(cell_bound_box[cell_idx].x_max - 
        cluster_bound_box_[merge_cluster_idx].x_max) < EPSILON) &&
        ((std::abs(cell_bound_box[cell_idx].y_min - 
        cluster_bound_box_[merge_cluster_idx].y_max) < EPSILON) ||
        std::abs(cell_bound_box[cell_idx].y_max - 
        cluster_bound_box_[merge_cluster_idx].y_min) < EPSILON);
    bool y_flag = 
        (std::abs(cell_bound_box[cell_idx].y_min - 
        cluster_bound_box_[merge_cluster_idx].y_min) < EPSILON) && 
        (std::abs(cell_bound_box[cell_idx].y_max - 
        cluster_bound_box_[merge_cluster_idx].y_max) < EPSILON) &&
        ((std::abs(cell_bound_box[cell_idx].x_min - 
        cluster_bound_box_[merge_cluster_idx].x_max) < EPSILON) ||
        std::abs(cell_bound_box[cell_idx].x_max - 
        cluster_bound_box_[merge_cluster_idx].x_min) < EPSILON);

    if (x_flag || y_flag){
        // std::cout << "cell idx:" << cell_idx << "\tcluster_idx:"
        //           << merge_cluster_idx << std::endl;
        return true;
    }
    return false;
}

std::size_t ReconstructionCluster::GridCluster(
    std::vector<int> &cell_cluster_map,
    std::vector<std::size_t> &cell_point_count,
    std::vector<std::unordered_map<image_t, ImageVisibility>> & cell_image_visib,
    std::vector<Box> &cell_bound_box,
    std::vector<std::size_t>& point_cell_map){
    std::size_t grid_slide = cell_bound_box.size();
    const int point_num = points_.size();
    for (std::size_t cell_idx = 0; cell_idx < grid_slide; cell_idx++){
        int num_filter = ImageVisibilityFilter(cell_image_visib[cell_idx], options_);        
        while(cell_image_visib[cell_idx].size() - num_filter > 
            options_.max_num_images * options_.max_num_images_factor && 
            options_.max_num_images > 0){
            if ((cell_bound_box[cell_idx].x_max - cell_bound_box[cell_idx].x_min) 
                < options_.min_cell_size &&
                (cell_bound_box[cell_idx].y_max - cell_bound_box[cell_idx].y_min) 
                < options_.min_cell_size){
                break;
            }

            cell_point_count.push_back(0);
            // point_cell_map.push_back(std::size_t());
            cell_image_visib.push_back(std::unordered_map<image_t, ImageVisibility> ());
            cell_bound_box.push_back(Box());
            if (((cell_bound_box[cell_idx].x_max - cell_bound_box[cell_idx].x_min) - 
                (cell_bound_box[cell_idx].y_max - cell_bound_box[cell_idx].y_min)) < 
                options_.max_cell_size * 0.01){
                // Split in Y direction
                float split_y = (cell_bound_box[cell_idx].y_min + 
                                cell_bound_box[cell_idx].y_max) / 2;
                Box box1 = cell_bound_box[cell_idx];
                Box box2 = cell_bound_box[cell_idx];
                box1.y_max = split_y;
                box2.y_min = split_y;
                cell_bound_box[cell_idx] = box1;
                cell_bound_box[grid_slide] = box2;
                
                for(std::size_t i = 0; i < point_num; ++i){
                    if (point_cell_map[i] != cell_idx){
                        continue;
                    }
                    auto &point = points_[i];
                    if (point.y() > split_y){
                        point_cell_map[i] = grid_slide;
                        cell_point_count[cell_idx]--;
                        cell_point_count[grid_slide]++;

                        for (auto image_id : point_visibility_[i]){
                            if (cell_image_visib[grid_slide].find(image_id) 
                                == cell_image_visib[grid_slide].end()){
                                cell_image_visib[grid_slide].emplace(image_id, 
                                ImageVisibility{image_id, 1, 
                                cell_image_visib[grid_slide][image_id].num_all_mappoint});
                            } else{
                                cell_image_visib[grid_slide][image_id].num_visib_mappoint++;
                            }

                            if(cell_image_visib[cell_idx][image_id].num_visib_mappoint > 1){
                                cell_image_visib[cell_idx][image_id].num_visib_mappoint--;
                            } else if (cell_image_visib[cell_idx].find(image_id) 
                                != cell_image_visib[cell_idx].end()){
                                cell_image_visib[cell_idx].erase(image_id);
                            }
                        }
                    }
                }
            } else {
                // Split in X direction
                float split_x = (cell_bound_box[cell_idx].x_min + 
                                cell_bound_box[cell_idx].x_max) / 2;
                Box box1 = cell_bound_box[cell_idx];
                Box box2 = cell_bound_box[cell_idx];
                box1.x_max = split_x;
                box2.x_min = split_x;
                cell_bound_box[cell_idx] = box1;
                cell_bound_box[grid_slide] = box2;
                
                for(std::size_t i = 0; i < point_num; ++i){
                    if (point_cell_map[i] != cell_idx){
                        continue;
                    }
                    auto &point = points_[i];
                    if (point.x() > split_x){
                        point_cell_map[i] = grid_slide;
                        cell_point_count[cell_idx]--;
                        cell_point_count[grid_slide]++;

                        for (auto image_id : point_visibility_[i]){
                            if (cell_image_visib[grid_slide].find(image_id) 
                                == cell_image_visib[grid_slide].end()){
                                cell_image_visib[grid_slide].emplace(image_id, 
                                ImageVisibility{image_id, 1, 
                                cell_image_visib[grid_slide][image_id].num_all_mappoint});
                            } else{
                                cell_image_visib[grid_slide][image_id].num_visib_mappoint++;
                            }

                            if(cell_image_visib[cell_idx][image_id].num_visib_mappoint > 1){
                                cell_image_visib[cell_idx][image_id].num_visib_mappoint--;
                            } else if (cell_image_visib[cell_idx].find(image_id) 
                                != cell_image_visib[cell_idx].end()){
                                cell_image_visib[cell_idx].erase(image_id);
                            }
                        }
                    }
                }
            }
            num_filter = ImageVisibilityFilter(cell_image_visib[cell_idx], options_);
            grid_slide++;
        }
        std::cout << "Cell "<< cell_idx << ": " << cell_image_visib[cell_idx].size() 
                  << " - " << num_filter << " = "<< cell_image_visib[cell_idx].size() - num_filter 
                  << "\tpoint size: " << cell_point_count[cell_idx] << std::endl;
    }

    std::size_t cell_num = cell_point_count.size();
    cell_cluster_map.resize(cell_num);
    std::fill(cell_cluster_map.begin(), cell_cluster_map.end(), -1);

    std::vector<std::pair<std::size_t, std::size_t> > dense_cells;
    dense_cells.reserve(cell_num);
    std::vector<std::size_t> sparse_cells;
    sparse_cells.reserve(cell_num);
    std::vector<unsigned char> cell_type_map(cell_num, 0);
    for (std::size_t i = 0; i < cell_num; i++) {
        // if (cell_point_count[i] < options_.min_pts_per_cluster) {
        // // if (cell_point_count[i] == 0) {
        //     cell_type_map[i] = 64;
        //     continue;
        // }

        if (cell_point_count[i] < options_.min_pts_per_cluster ||
            cell_image_visib[i].size() < options_.max_num_images 
            * options_.min_num_images_factor) {
            sparse_cells.push_back(i);
            cell_type_map[i] = 128;
            continue;
        }

        dense_cells.emplace_back(cell_point_count[i], i);
        cell_type_map[i] = 255;
    }
    dense_cells.shrink_to_fit();
    sparse_cells.shrink_to_fit();

    std::size_t cluster_idx = 0;
    if (dense_cells.empty()){
        dense_cells.emplace_back(cell_point_count[0], 0);
        sparse_cells.erase(sparse_cells.begin());
    }
    std::size_t dense_cell_num = dense_cells.size();
    cluster_bound_box_.resize(dense_cell_num);
    std::vector<std::size_t> cluster_point_count;
    std::vector<std::vector<std::size_t> > cluster_cells_map;
    std::vector<std::unordered_map<image_t, ImageVisibility>> cluster_image_visib;

    for (std::size_t i = 0; i < dense_cell_num; i++) {
        int cell_idx = dense_cells[i].second;
        if (cell_cluster_map[cell_idx] != -1) {
            continue;
        }

        std::size_t num_visited_points = 0;
        cluster_cells_map.push_back(std::vector<std::size_t>());
        auto &cluster_cells = cluster_cells_map.back();

        cell_cluster_map[cell_idx] = cluster_idx;
        cluster_cells.push_back(cell_idx);
        num_visited_points = cell_point_count[cell_idx];

        cluster_bound_box_[i] = cell_bound_box[cell_idx];

        cluster_point_count.push_back(num_visited_points);
        cluster_image_visib.push_back(cell_image_visib[cell_idx]);
        // std::cout << num_visited_points << " points clustered" << std::endl;
        cluster_idx++;
    }

    std::size_t cluster_num = cluster_idx;
    std::vector<std::size_t> cluster_idx_map;
    cluster_idx_map.reserve(cluster_num);
    for (std::size_t i = 0; i < cluster_num; i++) {
        cluster_idx_map.push_back(i);
    }
    cluster_idx_map.shrink_to_fit();

    // for (std::size_t i = 0; i < cluster_idx_map.size(); i++) {
    //     const auto &cluster_idx = cluster_idx_map[i];
    //     cluster_point_count[i] = cluster_point_count[cluster_idx];
    //     cluster_cells_map[i] = cluster_cells_map[cluster_idx];
    //     for (auto cell_idx : cluster_cells_map[i]) {
    //         cell_cluster_map[cell_idx] = i;
    //     }
    // }
    std::cout << "merge sparse clell" << std::endl;
    for (auto cell_idx : sparse_cells) {
        bool merge_flag = false;
        int merge_cluster_idx = -1;
        for (std::size_t i = 0; i < cluster_idx_map.size(); i++) {
            merge_cluster_idx = i;
            if (!WhetherMerge(cell_bound_box, cell_idx, merge_cluster_idx)) {
                continue;
            }

            std::unordered_map<image_t, ImageVisibility> temp_image_visib;
            temp_image_visib = cluster_image_visib[i];
            ImageVisibilityInsert(temp_image_visib, cell_image_visib[cell_idx]);
            const int num_filter = ImageVisibilityFilter(temp_image_visib, options_);
            if (temp_image_visib.size() - num_filter >= options_.max_num_images * 
                options_.max_num_images_factor){
                continue;
            }

            cell_cluster_map[cell_idx] = merge_cluster_idx;
            cluster_point_count[merge_cluster_idx] += cell_point_count[cell_idx];
            cluster_cells_map[merge_cluster_idx].push_back(cell_idx);
            cluster_bound_box_[merge_cluster_idx] += cell_bound_box[cell_idx];
            cell_type_map[cell_idx] = 255;
            cluster_image_visib[merge_cluster_idx].swap(temp_image_visib);
            merge_flag = true;
            std::cout << "merge: cell_idx: " << cell_idx << " to Cluster " << merge_cluster_idx << std::endl;
            break;
        }

        if (!merge_flag){
            std::size_t num_visited_points = 0;
            cluster_cells_map.push_back(std::vector<std::size_t>());
            auto &cluster_cells = cluster_cells_map.back();

            cell_cluster_map[cell_idx] = cluster_idx;
            cluster_cells.push_back(cell_idx);
            num_visited_points = cell_point_count[cell_idx];

            cluster_idx_map.push_back(cluster_idx);

            cluster_point_count.push_back(num_visited_points);
            cluster_image_visib.push_back(cell_image_visib[cell_idx]);
            cluster_bound_box_.push_back(cell_bound_box[cell_idx]);
            cell_type_map[cell_idx] = 255;
            // std::cout << num_visited_points << " points clustered" << std::endl;  

            cluster_idx++;
            cluster_num = cluster_idx;
        }
    }

    cluster_num = cluster_idx_map.size();
    cluster_point_count.resize(cluster_num);
    cluster_cells_map.resize(cluster_num);
    cluster_bound_box_.resize(cluster_num);

    return cluster_num;
}

std::size_t ReconstructionCluster::CommonViewCluster(
    std::vector<int> &cell_cluster_map,
    const std::vector<std::unordered_map<image_t, ImageVisibility>> cell_image_visib,
    const std::vector<Box> &cell_bound_box,
    const std::vector<std::size_t> &cell_point_count,
    const std::size_t grid_size_x,
    const std::size_t grid_size_y,
    const float valid_spacing) {
    const std::size_t grid_side = grid_size_x;
    const std::size_t grid_slide = grid_side * grid_size_y;
    static const int index_offset[3][2] = {{1, 0}, {0, 0}, {1, 1}};

    std::size_t cell_num = cell_point_count.size();
    cell_cluster_map.resize(cell_num);
    std::fill(cell_cluster_map.begin(), cell_cluster_map.end(), -1);
    // for (int i = 0; i < cell_num; i++){
    //     cell_cluster_map[i] = i;
    // }
    std::vector<unsigned char> cell_type_map(cell_num, 0);

    cluster_bound_box_.resize(cell_num);
    std::vector<std::vector<std::size_t> > cluster_cells_map;
    std::vector<std::unordered_map<image_t, ImageVisibility>> cluster_images_map;

    int cluster_num = 0;
    for (std::size_t cell_idx = 0; cell_idx < cell_num; cell_idx++){
        if (cell_type_map.at(cell_idx) == 255){
            continue;
        }

        cell_cluster_map[cell_idx] = cluster_num;
        cluster_bound_box_[cluster_num] = cell_bound_box[cell_idx];
        cluster_images_map.push_back(cell_image_visib[cell_idx]);

        cluster_cells_map.push_back(std::vector<std::size_t>());
        auto &cluster_cells = cluster_cells_map.back();
        cluster_cells.push_back(cell_idx);
        cell_type_map.at(cell_idx) = 255;

        while (true){
            // int cluster_id = cell_cluster_map.at(cell_idx);
            // auto& cluster_cells = cluster_cells_map.at(cluster_id);

            std::vector<int> x_cells;
            std::vector<int> y_cells;
            int min_x = grid_size_x;
            int min_y = grid_size_y;
            int max_x = 0;
            int max_y = 0;
            for (auto cell_id : cluster_cells){
                int x = cell_id % grid_side;
                int y = cell_id / grid_side;
                x_cells.push_back(x);
                y_cells.push_back(y);

                if (x < min_x){ min_x = x; }
                if (y < min_y){ min_y = y; }
                if (x > max_x){ max_x = x; }
                if (y > max_y){ max_y = y; }
            }

            int num_x = max_x - min_x + 1;
            int num_y = max_y - min_y + 1;

            // x neighbour
            std::vector<int> x_ner_cell_ids;
            int num_x_images = cluster_images_map[cluster_num].size();
            std::unordered_map<image_t, ImageVisibility> x_ner_images;
            x_ner_images.insert(cluster_images_map[cluster_num].begin(), 
                                cluster_images_map[cluster_num].end());
            bool x_ner_flag = false;
            for (int i = min_y; i <= max_y && (max_x + 1) < grid_size_x; i++){
                x_ner_flag = true;
                int temp_cell_idx = i * grid_side + max_x + 1;
                if (cell_type_map.at(temp_cell_idx) == 255){
                    x_ner_flag = false;
                    break;
                }
                x_ner_cell_ids.push_back(temp_cell_idx);
                num_x_images += cell_image_visib[temp_cell_idx].size();
                ImageVisibilityInsert(x_ner_images, cell_image_visib[temp_cell_idx]);
                const int num_filter = ImageVisibilityFilter(x_ner_images, options_);
                if (x_ner_images.size() - num_filter >= options_.max_num_images){
                    x_ner_flag = false;
                    // std::cout << cluster_num << "ã€€x image size: " << x_ner_images.size() 
                    //           << " num_filter: " << num_filter 
                    //           << " max_num_images: " << options_.max_num_images << std::endl;
                    break;
                }
            }

            //  y neighbour
            std::vector<int> y_ner_cell_ids;
            int num_y_images = cluster_images_map[cluster_num].size();
            std::unordered_map<image_t, ImageVisibility> y_ner_images;
            y_ner_images.insert(cluster_images_map[cluster_num].begin(), 
                                cluster_images_map[cluster_num].end());
            bool y_ner_flag = false;
            for (int i = min_x; i <= max_x && (max_y + 1) < grid_size_y; i++){
                y_ner_flag = true;
                int temp_cell_idx = (max_y + 1) * grid_side + i;
                if (cell_type_map.at(temp_cell_idx) == 255){
                    y_ner_flag = false;
                    break;
                }
                y_ner_cell_ids.push_back(temp_cell_idx);
                num_y_images += cell_image_visib[temp_cell_idx].size();
                ImageVisibilityInsert(y_ner_images, cell_image_visib[temp_cell_idx]);
                const int num_filter = ImageVisibilityFilter(y_ner_images, options_);
                if (y_ner_images.size() - num_filter >= options_.max_num_images){
                    y_ner_flag = false;
                    // std::cout << cluster_num << "ã€€y image size: " << y_ner_images.size() 
                    //           << " num_filter: " << num_filter 
                    //           << " max_num_images: " << options_.max_num_images << std::endl;
                    break;
                }
            }

            if(!(x_ner_flag || y_ner_flag)){
                break;
            }

            int x_repetition = x_ner_flag ? (x_ner_images.size() 
                               - cluster_images_map[cluster_num].size()) 
                               / x_ner_cell_ids.size() : MAX_INT;
            int y_repetition = y_ner_flag ? (y_ner_images.size() 
                               - cluster_images_map[cluster_num].size()) 
                               / y_ner_cell_ids.size() : MAX_INT;

            if (x_repetition < y_repetition){
                Box& temp_box = cluster_bound_box_[cluster_num];
                for (auto x_ner_cell_id : x_ner_cell_ids){
                    cell_cluster_map[x_ner_cell_id] = cluster_num;
                    cluster_cells.push_back(x_ner_cell_id);
                    temp_box += cell_bound_box[x_ner_cell_id];
                    cell_type_map.at(x_ner_cell_id) = 255;
                }
                cluster_images_map[cluster_num] = x_ner_images;
            }else{
                Box& temp_box = cluster_bound_box_[cluster_num];
                for (auto y_ner_cell_id : y_ner_cell_ids){
                    cell_cluster_map[y_ner_cell_id] = cluster_num;
                    cluster_cells.push_back(y_ner_cell_id);
                    temp_box += cell_bound_box[y_ner_cell_id];
                    cell_type_map.at(y_ner_cell_id) = 255;
                }
                cluster_images_map[cluster_num] = y_ner_images;
            }
        }
        cluster_num++;
    }

    return cluster_num;
}

void ReconstructionCluster::FilterImages(const std::shared_ptr<Reconstruction> clustered_reconstruction,
                  int& num_fiter_image, const Options options){
    int min_num_images = options.max_num_images / 5;
    const std::vector<image_t> image_ids = 
        clustered_reconstruction->RegisterImageIds();
    if (image_ids.size() < min_num_images){
        return;
    }
    std::vector<std::pair<image_t, double>> image_commonviews;
    for (auto image_id : image_ids){
        point2D_t cluster_image_mappoints 
            = clustered_reconstruction->Image(image_id).NumMapPoints();
        point2D_t ori_image_mappoints
            = reconstruction_->Image(image_id).NumMapPoints();
        double common_view = 
            float(cluster_image_mappoints)/float(ori_image_mappoints);
        image_commonviews.push_back(
            std::pair<image_t, double>(image_id, common_view));
    }

    std::sort(image_commonviews.begin(), image_commonviews.end(), 
            [=](std::pair<image_t, double>& a, std::pair<image_t, double>& b) 
            {return a.second < b.second; });
    
    uint64_t max_num_filter = image_commonviews.size() * options.max_filter_percent;
    bool bool_remove = false;
    if (image_commonviews.size() > options.max_num_images * options.max_num_images_factor){
        max_num_filter = std::max(max_num_filter, (uint64_t)(image_commonviews.size() - 
                              options.max_num_images * options.max_num_images_factor));
        bool_remove = true;
    }
    std::cout << "max_num_filter, image_commonviews.size(): " 
              << max_num_filter << ", " << image_commonviews.size() << std::endl;
    for (int cv_id = 0; cv_id < max_num_filter; cv_id++){
        if (image_ids.size() - num_fiter_image < min_num_images){
            return;
        }
        if (image_commonviews.at(cv_id).second < options.min_common_view ||
            (bool_remove && image_ids.size() - num_fiter_image > options.max_num_images * options.max_num_images_factor)){
            clustered_reconstruction->DeRegisterImage(
                image_commonviews.at(cv_id).first);
            num_fiter_image ++ ;
            // std::cout << image_commonviews.at(cv_id).first << " " 
            //           << image_commonviews.at(cv_id).second << std::endl;
        }
    }

    // remove remote images
    if (options.dist_threshold < 0){
        return;
    }
    float mean_dist = 0;
    int num_dist = 0;
    std::unordered_map<image_t, float> image_dist;
    for(auto image_id : clustered_reconstruction->RegisterImageIds()){
        float mean_dist_perimage = 0;
        int num_mappoint_perimage = 0;
        const Image& image = clustered_reconstruction->Image(image_id);
        const Eigen::Vector3d image_pose = image.ProjectionCenter();
        for(auto point2d : image.Points2D()){
            if (!point2d.HasMapPoint()){
                continue;
            }
            const MapPoint& mappoint 
                = clustered_reconstruction->MapPoint(point2d.MapPointId());
            const Eigen::Vector3d mp_pose = mappoint.XYZ();

            // compute distance of image-mappoint
            mean_dist_perimage += (image_pose - mp_pose).norm();
            num_mappoint_perimage++;
        }
        mean_dist += mean_dist_perimage;
        num_dist += num_mappoint_perimage;
        mean_dist_perimage = mean_dist_perimage/num_mappoint_perimage;
        image_dist.emplace(std::pair<image_t, float>(image_id, mean_dist_perimage));
    }
    mean_dist = mean_dist / num_dist;
    for (auto image_id : clustered_reconstruction->RegisterImageIds()){
        if (image_ids.size() - num_fiter_image < min_num_images){
            return;
        }
        if (image_dist[image_id] > mean_dist * options.dist_threshold){
            clustered_reconstruction->DeRegisterImage(image_id);
            num_fiter_image ++ ;
        }
    }
    return;
}

void ReconstructionCluster::AddMappoints(const std::shared_ptr<Reconstruction> clustered_reconstruction,
                  int& num_add_mappoint){
    const std::vector<image_t> filtered_image_ids = 
        clustered_reconstruction->RegisterImageIds();
    for (auto image_id : filtered_image_ids){
        Image& cluster_image = clustered_reconstruction->Image(image_id);
        const Image& ori_image = reconstruction_->Image(image_id);
        for (point2D_t idx = 0; idx < ori_image.NumPoints2D(); idx++){
            if (ori_image.Point2D(idx).HasMapPoint() && 
                !cluster_image.Point2D(idx).HasMapPoint()){
                mappoint_t mappoint_id = ori_image.Point2D(idx).MapPointId();
                // cluster_image.SetMapPointForPoint2D(idx, mappoint_id);

                const MapPoint& map_point = 
                        reconstruction_->MapPoint(mappoint_id);
                
                if (clustered_reconstruction->ExistsMapPoint(mappoint_id)){
                    MapPoint& clus_map_point = 
                        clustered_reconstruction->MapPoint(mappoint_id);
                    clus_map_point.Track().AddElement(cluster_image.ImageId(), idx);
                    CHECK(!cluster_image.Point2D(idx).HasMapPoint());
                    cluster_image.SetMapPointForPoint2D(idx, mappoint_id);
                    CHECK_LE(cluster_image.NumMapPoints(), cluster_image.NumPoints2D());
                }else{
                    Track track;
                    track.AddElement(cluster_image.ImageId(), idx);
                    clustered_reconstruction->AddMapPointWithError(
                        mappoint_id, map_point.XYZ(), std::move(track), 
                        map_point.Error(), map_point.Color(), true);
                    num_add_mappoint++;
                }
            }
        }
    }
}

void ReconstructionCluster::OutputColorRecons(){
    std::cout << "=> Color Cluster: ";
    std::size_t num_points = reconstruction_->NumMapPoints();
    for(std::size_t i = 0; i < num_points; ++i){
        auto cluster_idx = point_cluster_map_[i];
        if (cluster_idx < 0){
            continue;
        }
        MapPoint& map_point = reconstruction_->MapPoint(point_idx_[i]);
        Eigen::Vector3ub color;
        GreyToColorMix(cluster_idx, color);
        map_point.SetColor(color);
    }
    auto cluster_path = JoinPaths(out_workspace_path_, 
        "Clustered-" + std::to_string(num_out_reconstructions_));
    // CreateDirIfNotExists(cluster_path);
    // reconstruction_->WriteReconstruction(cluster_path, true);
    WritePLY(reconstruction_, cluster_path + ".ply");
    std::cout << reconstruction_->NumImages() << " images & "
                << reconstruction_->NumMapPoints() << " mappoints be saved" << std::endl;
}

int ReconstructionCluster::SaveClusteredRect(){
    //cluster reconstruction
    int num_images_clustered = 0;
#if 0
    std::size_t num_points = point_cluster_map_.size();
    std::cout << "Build Clustered Reconstruction..." << std::endl;
    int cluster_idx = 0;
    for (int idx = 0; idx < cluster_num_; idx++){
        auto clustered_reconstruction = std::make_shared<Reconstruction>();        
        for(std::size_t i = 0; i < num_points; ++i){
            if (idx < 0 || idx != point_cluster_map_[i]){
                continue;
            }
            const MapPoint& map_point = reconstruction_->MapPoint(point_idx_[i]);
            const Track& track = map_point.Track();
            // add camera and image
            for (const auto& track_el : track.Elements()){                
                if (!clustered_reconstruction->ExistsImage(track_el.image_id)){
                    Image& image = reconstruction_->Image(track_el.image_id);
                    if (!clustered_reconstruction->ExistsCamera(image.CameraId())){
                        const Camera& camera = reconstruction_->Camera(image.CameraId());
                        clustered_reconstruction->AddCamera(camera);
                    }

                    //test
                    clustered_reconstruction->AddImageNohasMapPoint(image);
                }
            }
            // add point3d
            clustered_reconstruction->AddMapPointWithError(
                point_idx_[i], map_point.XYZ(), map_point.Track(), 
                map_point.Error(), map_point.Color(), true);
        }
        if (clustered_reconstruction->NumRegisterImages() < 2){
            continue;
        }
        std::cout << "Cluster #" << cluster_idx << ": ";

        // Filter weak common view
        int num_fiter_image = 0;
        int num_add_mappoint = 0;

        FilterImages(clustered_reconstruction, num_fiter_image, options_);
        std::cout << "filter "<< num_fiter_image << "images / ";

        //add point3D with images
        AddMappoints(clustered_reconstruction, num_add_mappoint);
        std::cout << "add " << num_add_mappoint << " mappoints\t";
#ifdef OLD_FILE_SYATEM
        // save reconstruction
        auto cluster_path = 
            JoinPaths(out_workspace_path_, std::to_string(cluster_idx + num_out_reconstructions_));
        CreateDirIfNotExists(cluster_path);
        clustered_reconstruction->WriteReconstruction(cluster_path, true);

        WriteBoundBoxText(JoinPaths(cluster_path , ROI_BOX_NAME), cluster_bound_box_[cluster_idx]);
#else
        auto cluster_path = 
            JoinPaths(out_rect_path_, std::to_string(cluster_idx));
        CreateDirIfNotExists(cluster_path);
        // clustered_reconstruction->WriteRegisteredImagesInfo(
        //     JoinPaths(cluster_path , RECT_CLUSTER_NAME));
        WriteBoundBoxText(JoinPaths(cluster_path , ROI_BOX_NAME), 
                        cluster_bound_box_[cluster_idx]);

#endif
        cluster_idx++;
        num_images_clustered += clustered_reconstruction->NumRegisterImages();
        std::cout  << clustered_reconstruction->NumRegisterImages() << " images & " 
                    << clustered_reconstruction->NumMapPoints() << " mappoints be saved" << std::endl;
    }
#endif

    {
        YAML::Node yaml_node;
        yaml_node["num_clusters" ] = cluster_num_;

        auto rot = cluster_bound_box_[0].rot;
        std::vector<float> vec_rot(9);
        for(int i = 0; i < 3; i++){
            for (int j = 0; j < 3; j++){
                vec_rot[i*3 + j] = rot(i,j);
            }
        }
        yaml_node["transformation"] = vec_rot;

        yaml_node["box"]["x_min"] = prior_box_.x_box_min;
        yaml_node["box"]["y_min"] = prior_box_.y_box_min;
        yaml_node["box"]["z_min"] = prior_box_.z_box_min;
        yaml_node["box"]["x_max"] = prior_box_.x_box_max;
        yaml_node["box"]["y_max"] = prior_box_.y_box_max;
        yaml_node["box"]["z_max"] = prior_box_.z_box_max;

        for (int idx = 0; idx < cluster_num_; idx++){
            yaml_node[std::to_string(idx) ]["x_min"] = cluster_bound_box_[idx].x_min;
            yaml_node[std::to_string(idx) ]["y_min"] = cluster_bound_box_[idx].y_min;
            yaml_node[std::to_string(idx) ]["z_min"] = cluster_bound_box_[idx].z_min;
            yaml_node[std::to_string(idx) ]["x_max"] = cluster_bound_box_[idx].x_max;
            yaml_node[std::to_string(idx) ]["y_max"] = cluster_bound_box_[idx].y_max;
            yaml_node[std::to_string(idx) ]["z_max"] = cluster_bound_box_[idx].z_max;
        }

        std::string path = JoinPaths(out_rect_path_, DENSE_DIR, BOX_YAML);
        std::ofstream file;
        file.open(path);
        if (file.is_open()){
            file << yaml_node;
            file.close();
            std::cout << "Save Yaml: "  << path << std::endl;
        } else {
            std::cout << "Error: Yaml File is not Open!" << std::endl;
        }
    }

    // print color points with cluster_idx
    OutputColorRecons();
    return num_images_clustered;
}

}  // namespace mvs
}  // namespace sensemap