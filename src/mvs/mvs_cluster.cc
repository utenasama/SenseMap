// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include "mvs/mvs_cluster.h"
#include "util/common.h"
#include "util/cuda.h"
#include "util/cudacc.h"
#include "mvs/utils.h"

#include "yaml-cpp/yaml.h"

namespace sensemap {
namespace mvs {

void MVSCluster::GetGpuProp() {
    auto gpu_indices = CSVToVector<int>(options_.ConverToPM().gpu_index);
    if (gpu_indices.size() == 1 && gpu_indices[0] == -1) { 
        const int num_cuda_devices = GetNumCudaDevices();
        CHECK_GT(num_cuda_devices, 0);
        gpu_indices.resize(num_cuda_devices);
        for(int i = 0; i < num_cuda_devices; ++i) {
            gpu_indices[i] = i;
        }
    }
    const float G_byte = 1024 * 1024 * 1024;

    int max_gpu_index = 0;
    for (int i = 0; i < gpu_indices.size(); ++i) {
        max_gpu_index = std::max(max_gpu_index, gpu_indices.at(i));
    }

    max_gpu_memory_array_.resize(max_gpu_index + 1, 0);
    max_gpu_cudacore_array_.resize(max_gpu_index + 1, 0);
    max_gpu_texture_layered_0_.resize(max_gpu_index + 1, 0);
    max_gpu_texture_layered_1_.resize(max_gpu_index + 1, 0);
    max_gpu_texture_layered_2_.resize(max_gpu_index + 1, 0);
    for (int i = 0; i < gpu_indices.size(); i++){
        int idx = gpu_indices[i];
        cudaDeviceProp device;
        GetDeviceProp(idx, device);

        size_t gpu_free_size, gpu_total_size;
        CUDA_SAFE_CALL(cudaMemGetInfo(&gpu_free_size, &gpu_total_size));
        printf("Avail/Total Gpu Memory %f/%f\n", gpu_free_size / G_byte, gpu_total_size / G_byte);
        
        float available_gpu_memory = gpu_free_size / G_byte;
        if (options_.ConverToPM().max_gpu_memory < 0 || 
             options_.ConverToPM().max_gpu_memory > available_gpu_memory) {
            max_gpu_memory_array_[idx] = available_gpu_memory;
        } else {
            max_gpu_memory_array_[idx] = options_.ConverToPM().max_gpu_memory;
        }
        max_gpu_cudacore_array_[idx] = ConvertSMVer2Cores(device.major, device.minor) * device.multiProcessorCount;
        max_gpu_texture_layered_0_[idx] = device.maxTexture2DLayered[0];
        max_gpu_texture_layered_1_[idx] = device.maxTexture2DLayered[1];
        max_gpu_texture_layered_2_[idx] = device.maxTexture2DLayered[2];
        
        printf("Option: GPU#%d\n", idx);
        printf("        Global Memory = %f(G)\n", max_gpu_memory_array_[idx]);
        printf("        Max Layered Texture 2D Size =(%d,%d) x %d\n",
            max_gpu_texture_layered_0_[idx], max_gpu_texture_layered_1_[idx], max_gpu_texture_layered_2_[idx]);
        printf("        Stream Processor Size = %d\n", device.multiProcessorCount);
        printf("        CUDA Cores = %d\n", max_gpu_cudacore_array_[idx]);
    }
}

int MVSCluster::EstimatePatchMatchThreadsPerGPU() {
    const float G_byte = 1024 * 1024 * 1024;

    const auto& dense_reconstruction_path = JoinPaths(reconstruction_path_, 
                                                      DENSE_DIR, SPARSE_DIR);
    std::cout << dense_reconstruction_path << std::endl;
    Reconstruction reconstruction;
    reconstruction.ReadBinary(dense_reconstruction_path);

    size_t max_width = 0, max_height = 0;
    for (auto image_id : reconstruction.RegisterImageIds()) {
        class sensemap::Image image = reconstruction.Image(image_id);
        class sensemap::Camera camera = reconstruction.Camera(image.CameraId());
        max_width = std::max(max_width, camera.Width());
        max_height = std::max(max_height, camera.Height());
    }

    float total_byte = (94 + 5 * options_.ConverToPM().max_num_src_images);
    if (options_.ConverToPM().geom_consistency) {
        total_byte += 4 * options_.ConverToPM().max_num_src_images;
    }
    if (options_.ConverToPM().refine_with_semantic) {
        total_byte += (1 + options_.ConverToPM().max_num_src_images);
    }
    // if (options_.ConverToPM().has_prior_depth) {
    //     total_byte += 4;
    // }
    float used_gpu_memory = 1.4 * (max_width * max_height / G_byte) * total_byte;

    printf("Used GPU Memory: %f\n", used_gpu_memory);
    
    int num_tasks = 0;
    auto gpu_indices = CSVToVector<int>(options_.ConverToPM().gpu_index);
    std::cout << "gpu_index: " << options_.ConverToPM().gpu_index << " " << gpu_indices.size() << " " << gpu_indices[0] << std::endl;
    if (gpu_indices.size() == 1 && gpu_indices[0] == -1) {
        const int num_cuda_devices = GetNumCudaDevices();
        CHECK_GT(num_cuda_devices, 0);
        threads_per_gpu_.resize(num_cuda_devices, 0);
        for (int gpu_idx = 0; gpu_idx < num_cuda_devices; ++gpu_idx) {
            threads_per_gpu_[gpu_idx] = std::min(int(max_gpu_memory_array_[gpu_idx] / used_gpu_memory), MAX_THREADS_PER_GPU);
            num_tasks += threads_per_gpu_[gpu_idx];
            printf("GPU#%d: %d threads for patch match\n", gpu_idx, threads_per_gpu_[gpu_idx]);
        }
    } else {
        int max_gpu_index = 0;
        for (auto gpu_idx : gpu_indices) {
            max_gpu_index = std::max(gpu_idx, max_gpu_index);
        }
        threads_per_gpu_.resize(max_gpu_index + 1, 0);
        for (auto gpu_idx : gpu_indices) {
            threads_per_gpu_[gpu_idx] = std::min(int(max_gpu_memory_array_[gpu_idx] / used_gpu_memory), MAX_THREADS_PER_GPU);
            num_tasks += threads_per_gpu_[gpu_idx];
            printf("GPU#%d: %d threads for patch match\n", gpu_idx, threads_per_gpu_[gpu_idx]);
        }
    }
    return num_tasks;
}

MVSCluster::MVSCluster()
    : reconstruction_path_(""),
    num_out_reconstructions_(0){

        std::cout << "MVSCluster Empty: " << reconstruction_path_ << std::endl;
};

MVSCluster::MVSCluster(Options& options, const std::string& reconstruction_path)
    :options_(options),
    reconstruction_path_(reconstruction_path),
     num_out_reconstructions_(0){
         std::cout << "MVSCluster: " << reconstruction_path_ << std::endl;
};

void MVSCluster::Run() {
    const auto& dense_reconstruction_path = 
        JoinPaths(reconstruction_path_, DENSE_DIR);
    if (!ExistsDir(dense_reconstruction_path)) {
        return;
    }

    const auto& undistort_image_path = 
        JoinPaths(dense_reconstruction_path, IMAGES_DIR);
    if (!ExistsDir(undistort_image_path)) {
        return;
    }

    const auto& undistort_sparse_path = 
        JoinPaths(dense_reconstruction_path, SPARSE_DIR);
    if (!ExistsDir(undistort_sparse_path)) {
        return;
    }

    bool has_mask = false;
    const auto& undistort_mask_path = 
        JoinPaths(dense_reconstruction_path, MASKS_DIR);
    if (ExistsDir(undistort_mask_path)) {
        has_mask = true;
    }

    std::cout << "Reading workspace..." << std::endl;
    Workspace::Options workspace_options;

    workspace_options.image_as_rgb = false;
    // workspace_options.cache_size = options_.cache_size;
    workspace_options.cache_size = 1e-6;
    workspace_options.image_path = undistort_image_path;
    workspace_options.workspace_path = dense_reconstruction_path;
    workspace_options.workspace_format = "perspective";
    workspace_options.input_type = PHOTOMETRIC_TYPE;
    workspace_.reset(new Workspace(workspace_options));
    Model model = workspace_->GetModel();

    std::string update_path = JoinPaths(undistort_sparse_path, "update_images.txt");
    std::unordered_set<int> unique_image_idxs;
    if (options_.map_update && ExistsFile(update_path)){
        model.GetUpdateImageidxs(update_path, unique_image_idxs);
    } else {
        for (int i = 0; i < model.images.size(); i++){
            unique_image_idxs.insert(i);
        }
    }

    // Patch Match Image Cluster
    {
        PrintHeading2(StringPrintf("Patch Match Image Cluster"));
        std::cout << "Compute PM OverlappingImages" << std::endl;
        
        const auto pm_options = options_.ConverToPM();
        const size_t kCheckNum = pm_options.max_num_src_images * 2;
        const size_t kCheckNum_g = pm_options.max_num_src_images * 5;
        const double kMinTriangulationAngle = pm_options.min_triangulation_angle;
        std::vector<std::vector<int> > overlapping_images = 
        model.GetMaxOverlappingImages(kCheckNum_g, kMinTriangulationAngle, unique_image_idxs);

        std::vector<std::vector<int> > overlapping_cluster_images = overlapping_images;
        for (size_t i = 0; i < overlapping_images.size(); ++i) {
            const size_t eff_num_images = 
                std::min(overlapping_images[i].size(), kCheckNum);
            overlapping_cluster_images[i].resize(eff_num_images);
            overlapping_cluster_images[i].shrink_to_fit();
        }
        
        std::vector<std::vector<int>> cluster_image_map;
        uint64_t max_images_num;

        GetGpuProp();
        int num_thread = EstimatePatchMatchThreadsPerGPU();

        int num_cluster = PatchMatchImageCluster(pm_options, 
                cluster_image_map, max_images_num, num_thread, model.images, 
                overlapping_cluster_images, unique_image_idxs, has_mask);

        SaveImageClusterYaml(
            JoinPaths(dense_reconstruction_path, PATCH_MATCH_CLUATER_YAML) , 
            options_.cluster_step, cluster_image_map);
    }

    // Fusion Cluster Image Cluster
    {
        PrintHeading2(StringPrintf("Fusion Image Cluster"));
        std::cout << "Compute Fusion OverlappingImages" << std::endl;
        const double kMinTriangulationAngle = 0;
        std::vector<std::vector<int> > overlapping_images = 
        model.GetMaxOverlappingImages(
                options_.ConverToFusion().check_num_images, 
                kMinTriangulationAngle, unique_image_idxs);

        std::vector<std::vector<int>> cluster_image_map;
        std::vector<int> common_image_ids;
        uint64_t max_images_num;

        int num_cluster = FusionImageCluster(options_.ConverToFusion(),
            cluster_image_map, common_image_ids, max_images_num, 
            model.images, overlapping_images, unique_image_idxs);

        SaveImageClusterYaml(
            JoinPaths(dense_reconstruction_path, FUSION_CLUATER_YAML) , 
            options_.cluster_step, cluster_image_map, common_image_ids);
    }

    return;
}

bool MVSCluster::GetRoiBox(const PatchMatchOptions& options,
                                                         const Model& model,
                                                         const std::string& ori_box_path, 
                                                         Box& roi_box){

    if (ExistsFile(ori_box_path)){
        ReadBoundBoxText(ori_box_path, roi_box);
        float roi_box_factor = options.roi_box_factor;
        if (options.roi_box_width < 0 && options.roi_box_factor < 0){
            roi_box_factor = 0.01;
        }
        roi_box.SetBoundary(options.roi_box_width * 2, roi_box_factor * 2);
        roi_box.z_box_min = -FLT_MAX;
        roi_box.z_box_max = FLT_MAX;
        std::cout << StringPrintf("ROI(box.txt): [%f %f %f] -> [%f %f %f]\n", 
            roi_box.x_box_min, roi_box.y_box_min, roi_box.z_box_min,
            roi_box.x_box_max, roi_box.y_box_max, roi_box.z_box_max);
    } else {
        std::vector<Eigen::Vector3f> points;
        for (const Model::Point & point : model.points) {
            Eigen::Vector3f p(&point.x);
            points.emplace_back(p);
        }
        for (const mvs::Image & image : model.images) {
            Eigen::Vector3f C(image.GetC());
            points.emplace_back(C);
        }
        roi_box.x_min = roi_box.y_min = roi_box.z_min = FLT_MAX;
        roi_box.x_max = roi_box.y_max = roi_box.z_max = -FLT_MAX;
        for (auto point : points) {
            roi_box.x_min = std::min(roi_box.x_min, point.x());
            roi_box.y_min = std::min(roi_box.y_min, point.y());
            roi_box.z_min = std::min(roi_box.z_min, point.z());
            roi_box.x_max = std::max(roi_box.x_max, point.x());
            roi_box.y_max = std::max(roi_box.y_max, point.y());
            roi_box.z_max = std::max(roi_box.z_max, point.z());
        }
        float x_offset = (roi_box.x_max - roi_box.x_min) * 0.05;
        float y_offset = (roi_box.y_max - roi_box.y_min) * 0.05;
        float z_offset = (roi_box.z_max - roi_box.z_min) * 0.05;
        roi_box.x_box_min = roi_box.x_min - x_offset;
        roi_box.x_box_max = roi_box.x_max + x_offset;
        roi_box.y_box_min = roi_box.y_min - y_offset;
        roi_box.y_box_max = roi_box.y_max + y_offset;
        roi_box.z_box_min = roi_box.z_min - z_offset;
        roi_box.z_box_max = roi_box.z_max + z_offset;
        roi_box.rot = Eigen::Matrix3f::Identity();
        std::cout << StringPrintf("ROI(sparse): [%f %f %f] -> [%f %f %f]\n", 
            roi_box.x_box_min, roi_box.y_box_min, roi_box.z_box_min,
            roi_box.x_box_max, roi_box.y_box_max, roi_box.z_box_max);
    }
    return true;
}
#if 0
int MVSCluster::PatchMatchImageCluster(
                const PatchMatchOptions& options,
                 std::vector<std::vector<int>> &cluster_image_map,
                 uint64_t& max_images_num, const int num_thread,
                 const std::vector<mvs::Image> &images, 
                 const std::vector<std::vector<int> >& overlapping_images_){
    float geo_cf_ram_factor = 1;
    if (options.geom_consistency &&( (options.filter && options.geo_filter) || 
        (options.num_iter_geom_consistency > 1 && options.pyramid_max_level == 0))){
        geo_cf_ram_factor = 0.54f;
    } else if (options.filter && options.geo_filter) {
        geo_cf_ram_factor = 0.826f;
    }
    float max_ram = options.max_ram * geo_cf_ram_factor;

    const int all_image_num = images.size();
    if (max_ram < 0){
      cluster_image_map.resize(1);
      for (int i = 0; i < all_image_num; i++){
        cluster_image_map[0].push_back(i);
      }
      std::cout << "ImageCluster max_ram: " << max_ram << std::endl;
      return 1;
    }
    
    // compute max_num_images 
    float width = images.at(0).GetWidth();
    float height = images.at(0).GetHeight();
    for (int i = 0; i < images.size(); i++){
      float tmp_width = images.at(i).GetWidth();
      float tmp_height = images.at(i).GetHeight();
      if (width * height < tmp_width * tmp_height){
        width = tmp_width;
        height = tmp_height;
      }
    }
    // width * height * float size * (image + depth_map + normal_map )
    //  * coefficient_of_fluctuation
    float image_memory;
    if (options.refine_with_semantic) {
        image_memory = width * height * (1 + 4 + 12 + 1);
    } else {
        image_memory = width * height * (1 + 4 + 12);
    }
    const uint64_t G_byte = 1.0e9;
    const int idea_max_images = G_byte * max_ram / image_memory;
    if (idea_max_images >= num_thread * 2){
        max_images_num = (idea_max_images - num_thread) * 0.7;
    } else {
        max_images_num = idea_max_images * 0.5 * 0.7;
    }
    if (max_images_num <= 2) {
        std::cout << "Warning! image size is extremely huge!" << std::endl;
        std::cout << "image size: " << width << " " << height << std::endl;
    }
    max_images_num = std::max(max_images_num, (uint64_t)2);
    int num_cluster = all_image_num / max_images_num + 1;
    // num_cluster = 2;
    const int fixed_size = all_image_num / num_cluster + 1;
    num_cluster = std::ceil((float)all_image_num / fixed_size);

    std::cout << "Max_Ram: " << max_ram
              << "(G)\twidth * height: " <<  width * height
              << "\tNum_max_images: " << int(max_images_num) 
              << "\tNum_all_images: " << int(all_image_num) 
              << "\tNum_cluster: " << num_cluster << std::endl;
    cluster_image_map.resize(num_cluster);

    if (num_cluster == 1){
      for (int i = 0; i < all_image_num; i++){
        cluster_image_map[0].push_back(i);
      }
      return 1;
    }
    
    KMeans kmeans;
    for (int image_idx = 0; image_idx < all_image_num; image_idx++){
      auto& image = images.at(image_idx);
      Eigen::Vector3f location = Eigen::Map<const Eigen::Vector3f>(image.GetC());
      Tuple tuple;
      tuple.location = location.cast <double> ();
      tuple.id = image_idx;
      tuple.name = std::to_string(image_idx);
      kmeans.mv_pntcloud.emplace_back(tuple);
    }
    kmeans.SetK(num_cluster);
    kmeans.max_point_size = max_images_num;
    kmeans.fixed_size = fixed_size;
    // kmeans.SameSizeCluster();
    kmeans.SameSizeClusterWithConnection(overlapping_images_);

    cluster_image_map.resize(kmeans.m_k);
    for (size_t i = 0; i < kmeans.m_k; i++) {
      for (size_t j = 0; j < kmeans.m_grp_pntcloud[i].size(); j++){
        cluster_image_map[i].push_back(kmeans.m_grp_pntcloud[i].at(j).id);
      }
    }

#ifdef CLUSTER_INFO
    cv::RNG rng(12345);
    std::vector<PlyPoint> locations;
    std::vector<PlyPoint> common;

    std::vector<Eigen::Vector3i> color_k(kmeans.m_k);
    for (size_t i = 0; i < kmeans.m_k; i++) {
        int r_color = rng.uniform(0, 255);
        int g_color = rng.uniform(0, 255);
        int b_color = rng.uniform(0, 255);

        color_k[i][0] = r_color;
        color_k[i][1] = g_color;
        color_k[i][2] = b_color;

        for (size_t j = 0; j < kmeans.m_grp_pntcloud[i].size(); j++) {
            PlyPoint point;
            point.x = float(kmeans.m_grp_pntcloud[i][j].location[0]);
            point.y = float(kmeans.m_grp_pntcloud[i][j].location[1]);
            point.z = float(kmeans.m_grp_pntcloud[i][j].location[2]);
            point.r = r_color;
            point.g = g_color;
            point.b = b_color;
            Eigen::Vector3f cam_ray(images.at(kmeans.m_grp_pntcloud[i][j].id).GetViewingDirection());
            point.nx = cam_ray.x();
            point.ny = cam_ray.y();
            point.nz = cam_ray.z();
            locations.emplace_back(point);
        }
    }

    std::string SavePlyPath = "./locations-pm.ply";
    sensemap::WriteBinaryPlyPoints(SavePlyPath, locations, false, true);
    std::cout << "WriteBinaryPlyPoints :" << SavePlyPath << std::endl;
#endif

    std::cout << "Num_max_images: " << int(max_images_num) 
              << "\tNum_all_images: " << int(all_image_num) 
              << "\tNum_cluster: " <<  kmeans.m_k << std::endl;

    return kmeans.m_k;
}
#else
int MVSCluster::PatchMatchImageCluster(
                const PatchMatchOptions& options,
                 std::vector<std::vector<int>> &cluster_image_map,
                 uint64_t& max_images_num, const int num_thread,
                 const std::vector<mvs::Image> &images, 
                 const std::vector<std::vector<int> >& overlapping_images_,
                 const std::unordered_set<int>& update_image_idxs,
                 const bool has_mask){
    
    bool map_update = !update_image_idxs.empty();
    
    float geo_cf_ram_factor = 1;
    if (options.init_from_rgbd || options.image_type.compare("rgbd") == 0){
        geo_cf_ram_factor = 0.54f;
    } else if (options.geom_consistency &&( (options.filter && options.geo_filter) || 
        options.num_iter_geom_consistency > 1)){
        geo_cf_ram_factor = 0.54f;
    } else if (options.filter && options.geo_filter) {
        geo_cf_ram_factor = 0.826f;
    }
    float max_ram = options.max_ram * geo_cf_ram_factor;

    const int all_image_num = images.size();
    const int num_update_image = update_image_idxs.size();
    if (max_ram < 0){
      cluster_image_map.resize(1);
      for (int i = 0; i < all_image_num; i++){
        if (map_update && update_image_idxs.find(i) == update_image_idxs.end()){
            continue;
        }
        cluster_image_map[0].push_back(i);
      }
      std::cout << "ImageCluster 1 - cluster, MaxRam: " << max_ram << std::endl;
      return 1;
    }
    
    // compute max_num_images 
    Eigen::Vector3f image_center(0, 0, 0);
    size_t num_clusterd_image = 0;
    float width = images.at(0).GetWidth();
    float height = images.at(0).GetHeight();
    for (int i = 0; i < images.size(); i++){
      if (map_update && update_image_idxs.find(i) == update_image_idxs.end()){
        continue;
      }
      float tmp_width = images.at(i).GetWidth();
      float tmp_height = images.at(i).GetHeight();
      if (width * height < tmp_width * tmp_height){
        width = tmp_width;
        height = tmp_height;
      }
      const float *C = images.at(i).GetC();
      const Eigen::Vector3f image_c(C[0], C[1], C[2]);
      image_center += image_c;
      num_clusterd_image++;
    }
    image_center = image_center / (float)num_clusterd_image;

    // width * height * float size * (image + depth_map + normal_map )
    //  * coefficient_of_fluctuation
    float image_memory;
    if (options.refine_with_semantic) {
        image_memory = width * height * (1 + 4 + 12 + 1);
    } else {
        image_memory = width * height * (1 + 4 + 12);
    }
    if (has_mask) {
        image_memory += width * height;
    }
    const uint64_t G_byte = 1.0e9;
    const int idea_max_images = G_byte * max_ram / image_memory;
    if (idea_max_images >= num_thread * 2){
        max_images_num = (idea_max_images - num_thread) * 0.7;
    } else {
        max_images_num = idea_max_images * 0.5 * 0.7;
    }
    if (max_images_num <= 2) {
        std::cout << "Warning! image size is extremely huge!" << std::endl;
        std::cout << "image size: " << width << " " << height << std::endl;
    }
    max_images_num = std::max(max_images_num, (uint64_t)2);
    int num_cluster = num_clusterd_image / max_images_num + 1;
    // num_cluster = 2;
    const int fixed_size = num_clusterd_image / num_cluster + 1;
    num_cluster = std::ceil((float)num_clusterd_image / fixed_size);

    std::cout << "map_update: " << map_update << "\tMax_Ram: " << max_ram
              << "(G)\twidth * height: " <<  width * height
              << "\tNum_max_images: " << int(max_images_num) 
              << "\tNum_clustered_images: " << int(num_clusterd_image) 
              << "\tNum_all_images: " << int(all_image_num) 
              << "\tNum_cluster: " << num_cluster 
              << "\tFix_size: " << fixed_size << std::endl;
    // cluster_image_map.resize(num_cluster);

    if (num_cluster == 1){
        cluster_image_map.resize(num_cluster);
      for (int i = 0; i < all_image_num; i++){
        if (map_update && update_image_idxs.find(i) == update_image_idxs.end()){
            continue;
        }
        cluster_image_map[0].push_back(i);
      }
      return 1;
    }
    
    std::unordered_map<int, std::vector<int>> src_2_refs_map;
    std::unordered_map<int, std::vector<int>> ref_2_srcs_map;
    for (int i = 0; i < images.size(); i++){
        const size_t eff_max_num_src_images = std::min(
            overlapping_images_[i].size(), options_.max_num_src_images);
        for (int j = 0; j < eff_max_num_src_images; j++){
            ref_2_srcs_map[i].push_back(overlapping_images_[i][j]);
            src_2_refs_map[overlapping_images_[i][j]].push_back(i);
        }
    }

    std::unordered_map<int, float> src_scores;
    std::vector<float > v_distances(images.size(), 0.f);
    int init_image_id = -1;
    float max_distance = -1;
    for (size_t i = 0; i < images.size(); i++){
        if (map_update && update_image_idxs.find(i) == update_image_idxs.end()){
            continue;
        }
        src_scores[i] = 0;
        const float *C = images.at(i).GetC();
        const Eigen::Vector3f image_C(C[0], C[1], C[2]);
        float distance = (image_C - image_center).norm();
        v_distances[i] = distance;
        
        if (distance > max_distance){
            init_image_id = i;
            max_distance = distance;
        }
    }
    std::cout << "src_scores size: " << src_scores.size() << std::endl;
    std::cout << "init_image_id: " << init_image_id << ", max_distance: " 
        << max_distance << std::endl;

    std::unordered_set<int > whole_ids;
    std::vector<std::unordered_set<int >> cluster_whole_ids;
    int next_ref_id = init_image_id;
    while (next_ref_id >= 0){
        whole_ids.insert(next_ref_id);
        src_scores.erase(next_ref_id);
        // for (const auto connect_ref_id : src_2_refs_map[next_ref_id]){
        for (int k = 0; k < src_2_refs_map[next_ref_id].size(); k++){
            const auto connect_ref_id = src_2_refs_map[next_ref_id].at(k);
            if (src_scores.find(connect_ref_id) != src_scores.end()){
                src_scores[connect_ref_id] += 
                    (1 + 0.02 * (src_2_refs_map[next_ref_id].size() - k));
            }
        }

        next_ref_id = -1;
        if (whole_ids.size() >= fixed_size || src_scores.empty()){
            // cluster_whole_ids.push_back(whole_ids);
            std::vector<int> v_cluster_ids;
            v_cluster_ids.insert(v_cluster_ids.end(), whole_ids.begin(), whole_ids.end());
            cluster_image_map.push_back(v_cluster_ids);

            float max_dist = -1;
            for(auto src_score : src_scores){
                if (map_update && update_image_idxs.find(src_score.first) == update_image_idxs.end()){
                    continue;
                }
                src_score.second = 0;
                if (v_distances[src_score.first] > max_dist){
                    max_dist = v_distances[src_score.first];
                    next_ref_id = src_score.first;
                }
            }
            whole_ids.clear();
            std::cout << "cluster image id, size, next_id: " << cluster_image_map.size() << ", "
                << v_cluster_ids.size() << ", " << next_ref_id << std::endl;

        } else {
            float max_score = -1;
            for (auto src_score : src_scores){
                if (map_update && update_image_idxs.find(src_score.first) == update_image_idxs.end()){
                    continue;
                }
                if (src_score.second > max_score){
                    max_score = src_score.second;
                    next_ref_id = src_score.first;
                }
            }
        }
    }


    std::cout << "Num_max_images: " << int(max_images_num) 
              << "\tNum_all_images: " << int(all_image_num) 
              << "\tNum_cluster: " << cluster_image_map.size() << std::endl;

    return cluster_image_map.size();
}

#endif
int MVSCluster::FusionImageCluster(
                const StereoFusion::Options& options,
                 std::vector<std::vector<int>> &cluster_image_map,
                 std::vector<int>& common_image_ids,
                 uint64_t& max_images_num,
                 const std::vector<mvs::Image> &images,
                 const std::vector<std::vector<int> >& overlapping_images_,
                 const std::unordered_set<int>& update_image_idxs){
    
    bool map_update = !update_image_idxs.empty();

    const int all_image_num = images.size();
    
    // compute max_num_images 
    int num_clusterd_image = 0;
    float width = images.at(0).GetWidth();
    float height = images.at(0).GetHeight();
    for (int i = 0; i < images.size(); i++){
      if (map_update && update_image_idxs.find(i) == update_image_idxs.end()){
        continue;
      }
      float tmp_width = images.at(i).GetWidth();
      float tmp_height = images.at(i).GetHeight();
      if (width * height < tmp_width * tmp_height){
        width = tmp_width;
        height = tmp_height;
      }
      num_clusterd_image++;
    }
    // width * height * float size * (image + depth_map + normal_map 
    // + semantic_map + mask) * coefficient_of_fluctuation
    float image_memory;
    if (options.with_normal) {
      image_memory = width * height * (3 + 4 + 12 + 1 + 1);
    } else {
      image_memory = width * height * (3 + 4 + 1 + 1);
    }
    uint64_t G_byte =  1.0e9;
    max_images_num = options.max_ram * G_byte 
                                        * options.ram_eff_factor/ image_memory 
                                        * (1 - 4 * options.fuse_common_persent); 
    int num_cluster = num_clusterd_image / max_images_num + 1;
    // int num_cluster = 2;
    const int fixed_size = num_clusterd_image / num_cluster + 1;
    num_cluster = std::ceil((float)num_clusterd_image / fixed_size);

    std::cout  << "Max_Ram: " << options.max_ram
              << "(G)\twidth * height: " <<  width * height
              << "\tram_eff_factor, fuse_common_persent: " 
              << options.ram_eff_factor << ", " << options.fuse_common_persent
              << "\tmax_width * height: " <<  width * height
              << "\tNum_max_images: " << int(max_images_num) 
              << "\tNum_clustered_images: " << int(num_clusterd_image) 
              << "\tNum_all_images: " << int(all_image_num) 
              << "\tNum_cluster: " << num_cluster << std::endl;
    cluster_image_map.resize(num_cluster);

    if (num_cluster == 1){
      for (int i = 0; i < all_image_num; i++){
        if (map_update && update_image_idxs.find(i) == update_image_idxs.end()){
            continue;
        }
        cluster_image_map[0].push_back(i);
      }
      return 1;
    }
    
    KMeans kmeans;
    for (int image_idx = 0; image_idx < all_image_num; image_idx++){
      if (map_update && update_image_idxs.find(image_idx) == update_image_idxs.end()){
        continue;
      }
      auto& image = images.at(image_idx);
      Eigen::Vector3f location = Eigen::Map<const Eigen::Vector3f>(image.GetC());
      Tuple tuple;
      tuple.location = location.cast <double> ();
      tuple.id = image_idx;
      tuple.name = std::to_string(image_idx);
      kmeans.mv_pntcloud.emplace_back(tuple);
    }
    kmeans.SetK(num_cluster);
    kmeans.max_point_size = max_images_num;
    kmeans.fixed_size = fixed_size;
    // kmeans.SameSizeCluster();
    kmeans.SameSizeClusterWithConnection(overlapping_images_);
    int common_images_size = std::min(
        kmeans.fixed_size * options.fuse_common_persent + 1, options.check_num_images/2.0f);

    std::vector<std::unordered_map<int, std::vector<Tuple>>> neighbors_points;
    std::vector<std::vector<int>> neighbors;
    kmeans.FindNeighborsAndCommonPointsWithConnection_AllPoints(neighbors_points, 
      neighbors, overlapping_images_);

    cluster_image_map.resize(kmeans.m_k);
    for (size_t i = 0; i < kmeans.m_k; i++) {
      for (size_t j = 0; j < kmeans.m_grp_pntcloud[i].size(); j++){
        cluster_image_map[i].push_back(kmeans.m_grp_pntcloud[i].at(j).id);
      }
      int num_neighbors = neighbors[i].size();
      common_images_size = (num_neighbors <= 4 ? common_images_size : 
        common_images_size * 4 / num_neighbors + 1);
      for (size_t j = 0; j < neighbors[i].size(); j++) {
        for (size_t k = 0; k < neighbors_points[i][neighbors[i][j]].size() 
            && k < common_images_size; k++) {
            cluster_image_map[neighbors[i][j]].push_back(
              neighbors_points[i][neighbors[i][j]][k].id);
            common_image_ids.push_back(
              neighbors_points[i][neighbors[i][j]][k].id);
        }
      }
    }
    std::set<int> st(common_image_ids.begin(), common_image_ids.end());
    common_image_ids.assign(st.begin(), st.end());

#ifdef CLUSTER_INFO
    cv::RNG rng(12345);
    std::vector<PlyPoint> locations;
    std::vector<PlyPoint> common;

    std::vector<Eigen::Vector3i> color_k(kmeans.m_k);
    for (size_t i = 0; i < kmeans.m_k; i++) {
        int r_color = rng.uniform(0, 255);
        int g_color = rng.uniform(0, 255);
        int b_color = rng.uniform(0, 255);

        color_k[i][0] = r_color;
        color_k[i][1] = g_color;
        color_k[i][2] = b_color;

        for (size_t j = 0; j < kmeans.m_grp_pntcloud[i].size(); j++) {
            PlyPoint point;
            point.x = float(kmeans.m_grp_pntcloud[i][j].location[0]);
            point.y = float(kmeans.m_grp_pntcloud[i][j].location[1]);
            point.z = float(kmeans.m_grp_pntcloud[i][j].location[2]);
            point.r = r_color;
            point.g = g_color;
            point.b = b_color;
            Eigen::Vector3f cam_ray(images.at(kmeans.m_grp_pntcloud[i][j].id).GetViewingDirection());
            point.nx = cam_ray.x();
            point.ny = cam_ray.y();
            point.nz = cam_ray.z();
            locations.emplace_back(point);
        }
    }

    for (size_t i = 0; i < kmeans.m_k; i++) {
        for (size_t j = 0; j < neighbors[i].size(); j++) {
            for (size_t k = 0; k < neighbors_points[i][neighbors[i][j]].size() 
                 && k < common_images_size; k++) {
                PlyPoint point;
                point.x = float(neighbors_points[i][neighbors[i][j]][k].location[0]);
                point.y = float(neighbors_points[i][neighbors[i][j]][k].location[1]);
                point.z = float(neighbors_points[i][neighbors[i][j]][k].location[2]);
                point.r = color_k[neighbors[i][j]][0];
                point.g = color_k[neighbors[i][j]][1];
                point.b = color_k[neighbors[i][j]][2];
                Eigen::Vector3f cam_ray(images.at(neighbors_points[i][neighbors[i][j]][k].id).GetViewingDirection());
                point.nx = cam_ray.x();
                point.ny = cam_ray.y();
                point.nz = cam_ray.z();
                common.emplace_back(point);
            }
        }
    }
    std::string SavePlyPath = "./locations-fu.ply";
    std::string SavePlyPath_common = "./locations_common-fu.ply";
    sensemap::WriteBinaryPlyPoints(SavePlyPath, locations, false, true);
    sensemap::WriteBinaryPlyPoints(SavePlyPath_common, common, false, true);
    std::cout << "WriteBinaryPlyPoints :" << SavePlyPath << std::endl;
#endif

    std::cout << "Num_max_images: " << int(max_images_num) 
              << "\tNum_all_images: " << int(all_image_num) 
              << "\tNum_cluster: " <<  kmeans.m_k << std::endl;

    return kmeans.m_k;
}


bool MVSCluster::SaveImageClusterYaml(std::string path, const int step,
    const std::vector<std::vector<int>> &cluster_image_map,
    const std::vector<int>& common_image_ids){
        YAML::Node yaml_node;
        yaml_node["num_blocks"] = std::ceil((float)cluster_image_map.size() / (float)step);
        yaml_node["num_clusters" ] = cluster_image_map.size();
        yaml_node["step" ] = step;
        for (int i = 0; i < cluster_image_map.size(); i++){
            yaml_node[std::to_string(i) ]["ids"]= cluster_image_map.at(i);
        }

        if (!common_image_ids.empty()){
            yaml_node["has_common"] = true;
            yaml_node["common_ids"] = common_image_ids;
        } else {
            yaml_node["has_common"] = false;
        }

        std::ofstream file;
        file.open(path);
        if (file.is_open()){
            file << yaml_node;
            file.close();
            std::cout << "Save Yaml: "  << path << std::endl;
        } else {
            std::cout << "Error: Yaml File is not Open!" << std::endl;
        }
        return true;
}

bool MVSCluster::ReadImageClusterYaml(const std::string path, 
    std::vector<std::vector<int>> &cluster_image_map, int & step){
    std::cout << "ReadImageClusterYaml" << std::endl;
    YAML::Node node = YAML::LoadFile(path);
    step = node["step"].as<int>();
    int num_cluster = node["num_clusters"].as<int>();
    cluster_image_map.resize(num_cluster);
    // std::cout << "num_cluster: " << num_cluster << std::endl;
    for (int i = 0; i < num_cluster; i++){
        for (YAML::const_iterator it= node[std::to_string(i)]["ids"].begin(); 
                it != node[std::to_string(i)]["ids"].end();++it){
            cluster_image_map.at(i).push_back(it->as<int>());
        }
    }

    std::cout << "num_cluster: " << num_cluster << std::endl;

    return true;
};


bool MVSCluster::ReadImageClusterYaml(const std::string path, 
    std::vector<std::vector<int>> &cluster_image_map,
    std::vector<int>& common_image_ids, int & step){
    std::cout << "ReadImageClusterYaml" << std::endl;
    YAML::Node node = YAML::LoadFile(path);
    step = node["step"].as<int>();
    int num_cluster = node["num_clusters"].as<int>();
    cluster_image_map.resize(num_cluster);
    // std::cout << "num_cluster: " << num_cluster << std::endl;
    for (int i = 0; i < num_cluster; i++){
        for (YAML::const_iterator it= node[std::to_string(i)]["ids"].begin(); 
                it != node[std::to_string(i)]["ids"].end();++it){
            cluster_image_map.at(i).push_back(it->as<int>());
        }
    }
    // std::cout << "read image cluster: " << num_cluster << std::endl;

    common_image_ids.clear();
    bool has_common = node["has_common"].as<bool>();
    if (has_common){
        for (YAML::const_iterator it= node["common_ids"].begin(); 
            it != node["common_ids"].end();++it){
                common_image_ids.push_back(it->as<int>());
        }
    }

    std::cout << "read image cluster / common_image: " 
        << num_cluster << ", " << common_image_ids.size() 
        << "(" << has_common << ")" << std::endl;

    // for (int i = 0; i < common_image_ids.size(); i++){
    //     std::cout << common_image_ids.at(i) << ", ";
    // }
    // std::cout << std::endl;

    return true;
};

}  // namespace mvs
}  // namespace sensemap