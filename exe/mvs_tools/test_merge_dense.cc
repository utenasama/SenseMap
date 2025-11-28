#include "base/reconstruction.h"
#include "base/common.h"
#include "util/exception_handler.h"
#include "util/misc.h"
#include "util/ply.h"
#include "mvs/model.h"
#include "mvs/workspace.h"

#include <opencv2/imgcodecs.hpp>

#include "../Configurator_yaml.h"
#include "base/version.h"

using namespace sensemap;

enum class RGBDConversion{
    None = 0,
    RGBD_To_Perspective,
    Perspective_To_RGBD,
};

Reconstruction MergeReconstructions(const Reconstruction & reconstruction1, const Reconstruction & reconstruction2, RGBDConversion conversion) {
    auto get_camera_string = [](const Camera & camera) {
        return camera.ModelName() + ", " + 
                std::to_string(camera.Width()) + ", " + 
                std::to_string(camera.Height()) + ", " + 
                camera.ParamsToString();
    };

    Reconstruction reconstruction3 = reconstruction1;
    std::unordered_map<std::string, image_t> existed_image_names;
    std::unordered_map<std::string, camera_t> existed_cameras;
    image_t max_image_id = 0;
    for (auto & image3 : reconstruction3.Images()) {
        image_t image3_id = image3.first;
        existed_image_names[image3.second.Name()] = image3_id;
        max_image_id = std::max(image3_id, max_image_id);
    }

    camera_t max_camera_id = 0;
    for (auto & camera3 : reconstruction3.Cameras()){
        camera_t camera3_id = camera3.first;
        std::string camera3_string = get_camera_string(reconstruction3.Camera(camera3_id));
        existed_cameras[camera3_string] = camera3_id;
        max_camera_id = std::max(camera3_id, max_camera_id);
    }
    
    std::unordered_map<mappoint_t, mappoint_t> map_old_new_mappoint_ids;
    for (auto & image2 : reconstruction2.Images()) {
        if (existed_image_names.count(image2.second.Name()) == 0) {
            camera_t camera2_id = image2.second.CameraId();
            const Camera & camera2 = reconstruction2.Camera(camera2_id);

            camera_t camera3_id = kInvalidCameraId;
            std::string camera2_string = get_camera_string(camera2);
            if (existed_cameras.count(camera2_string) == 0) {
                camera3_id = ++max_camera_id;
                Camera camera3 = camera2;
                camera3.SetCameraId(camera3_id);
                reconstruction3.AddCamera(camera3);

                existed_cameras[camera2_string] = camera3_id;
            } else {
                camera3_id = existed_cameras[camera2_string];
            }

            image_t image2_id = image2.first;
            image_t image3_id = ++max_image_id;
            {
                Image image3;
                image3.SetImageId(image3_id);
                image3.SetCameraId(camera3_id);
                image3.SetQvec(image2.second.Qvec());
                image3.SetTvec(image2.second.Tvec());
                image3.SetPoints2D(image2.second.Points2D());
                if (conversion == RGBDConversion::RGBD_To_Perspective) {
                    image3.SetName(image2.second.Name() + ".jpg");
                } else {
                    image3.SetName(image2.second.Name());
                }

                reconstruction3.AddImage(image3);
                reconstruction3.RegisterImage(image3.ImageId());
            }

            Image & image3 = reconstruction3.Image(image3_id);
            for (int point2D_idx = 0; point2D_idx < image2.second.Points2D().size(); point2D_idx++) {
                auto & point2D2 = image2.second.Point2D(point2D_idx);
                auto & point2D3 = image3.Point2D(point2D_idx);
                point2D3.SetMapPointId(kInvalidMapPointId);

                if (point2D2.HasMapPoint()) {
                    mappoint_t mappoint2_id = point2D2.MapPointId();
                    if (map_old_new_mappoint_ids.count(mappoint2_id) == 0) {
                        const MapPoint & mappoint2 = reconstruction2.MapPoint(mappoint2_id);
                        Track track3;
                        track3.AddElement(image3_id, point2D_idx);

                        mappoint_t mappoint3_id = reconstruction3.AddMapPoint(mappoint2.XYZ(), std::move(track3), mappoint2.Color());
                        map_old_new_mappoint_ids[mappoint2_id] = mappoint3_id;
                    } else {
                        mappoint_t mappoint3_id = map_old_new_mappoint_ids[mappoint2_id];
                        reconstruction3.AddObservation(mappoint3_id, TrackElement(image3_id, point2D_idx));
                    }
                }
            }
        }
    }

    std::cout << "1st reconstruction: " << reconstruction1.Images().size() << " images, " << reconstruction1.Cameras().size() << " cameras." << std::endl;
    std::cout << "2nd reconstruction: " << reconstruction2.Images().size() << " images, " << reconstruction2.Cameras().size() << " cameras." << std::endl;
    std::cout << "Fin reconstruction: " << reconstruction3.Images().size() << " images, " << reconstruction3.Cameras().size() << " cameras." << std::endl;
    return reconstruction3;
}

void MergeImagePaths(const std::string & image_path1, const std::string & image_path2, const std::string & image_path3, RGBDConversion conversion) {
    std::vector<std::string> image_list1;
    std::vector<std::string> image_list2;
    if (!image_path1.empty() && ExistsDir(image_path1)) {
        image_list1 = GetRecursiveFileList(image_path1);
    }
    if (!image_path2.empty() && ExistsDir(image_path2)) {
        image_list2 = GetRecursiveFileList(image_path2);
    }

    for (auto & image_file1 : image_list1) {
        std::string image_name1 = GetRelativePath(image_path1, image_file1);
        std::string image_file3 = JoinPaths(image_path3, image_name1);

        if (!boost::filesystem::exists(image_file3)) {
            if (!boost::filesystem::exists(GetParentDir(image_file3))) {
                boost::filesystem::create_directories(GetParentDir(image_file3));
            }
            boost::filesystem::copy_file(image_file1, image_file3);
        }
    }

    for (auto & image_file2 : image_list2) {
        std::string image_name2 = GetRelativePath(image_path2, image_file2);
        std::string image_file3 = JoinPaths(image_path3, image_name2);
        if (conversion == RGBDConversion::Perspective_To_RGBD) {
            if (boost::filesystem::path(image_file3).extension().string() == std::string(".") + MASK_EXT) {
                // is semantic
                image_file3 = image_path3.substr(0, image_path3.length() - 4) + ".jpg." + MASK_EXT;
            } else {
                // is image
                image_file3 += ".jpg";
            }
        }

        if (!boost::filesystem::exists(image_file3)) {
            if (!boost::filesystem::exists(GetParentDir(image_file3))) {
                boost::filesystem::create_directories(GetParentDir(image_file3));
            }
            boost::filesystem::copy_file(image_file2, image_file3);
        }
    }
}

void MergeFusedPlys(
    const std::string & dense_path1, const mvs::Model & model1, 
    const std::string & dense_path2, const mvs::Model & model2, 
    const std::string & dense_path3, const mvs::Model & model3,
    bool is_rgbd1
) {
    const std::string fused_path1 = JoinPaths(dense_path1, FUSION_NAME);
    const std::string fused_path2 = JoinPaths(dense_path2, FUSION_NAME);
    const std::string fused_path3 = JoinPaths(dense_path3, FUSION_NAME);
    const std::string mask_path1 = JoinPaths(dense_path1, CHANGE_MASKS_DIR);

    std::vector<uint32_t> map_idx1_to_idx3(model1.images.size());
    std::vector<uint32_t> map_idx2_to_idx3(model2.images.size());
    for (size_t i = 0; i < model1.images.size(); i++) {
        std::string image_name1 = model1.GetImageName(i);
        map_idx1_to_idx3[i] = model3.GetImageIdx(image_name1);
    }
    for (size_t i = 0; i < model2.images.size(); i++) {
        std::string image_name2 = model2.GetImageName(i);
        map_idx2_to_idx3[i] = model3.GetImageIdx(image_name2);
    }
    
    std::string fused_vis_path1 = fused_path1 + ".vis";
    std::string fused_vis_path2 = fused_path2 + ".vis";
    std::string fused_vis_path3 = fused_path3 + ".vis";
    std::string fused_wgt_path1 = fused_path1 + ".wgt";
    std::string fused_wgt_path2 = fused_path2 + ".wgt";
    std::string fused_wgt_path3 = fused_path3 + ".wgt";
    std::string fused_sem_path1 = fused_path1 + ".sem";
    std::string fused_sem_path2 = fused_path2 + ".sem";
    std::string fused_sem_path3 = fused_path3 + ".sem";
    CHECK(ExistsFile(fused_vis_path1) && ExistsFile(fused_vis_path2));
    
    std::vector<PlyPoint> fused_points1 = ReadPly(fused_path1);
    std::vector<PlyPoint> fused_points2 = ReadPly(fused_path2);
    std::vector<PlyPoint> fused_points3;
    std::vector<std::vector<uint32_t> > fused_points_vis1;
    std::vector<std::vector<uint32_t> > fused_points_vis2;
    std::vector<std::vector<uint32_t> > fused_points_vis3;
    ReadPointsVisibility(fused_vis_path1, fused_points_vis1);
    ReadPointsVisibility(fused_vis_path2, fused_points_vis2);

    bool has_sem = false;
    if (ExistsFile(fused_sem_path1)) {
        ReadPointsSemantic(fused_sem_path1, fused_points1);
        if (ExistsFile(fused_sem_path2)) {
            ReadPointsSemantic(fused_sem_path2, fused_points2);
        } else {
            std::cout << "Found " << fused_sem_path1 << ", "
                      << "but " << fused_sem_path2 << " is not available. " << std::endl;
        }
        has_sem = true;
    }

    std::vector<unsigned char> all_discard_flag1(fused_points1.size(), 0);
    std::vector<std::vector<unsigned char>> view_discard_flag1(fused_points1.size());
    if (!mask_path1.empty() && ExistsDir(mask_path1)) {
        std::unordered_map<uint32_t, cv::Mat> map_idx1_to_mask;
        for (size_t i = 0; i < model1.images.size(); i++) {
            std::string mask_file1 = JoinPaths(mask_path1, model1.GetImageName(i)) + "." + MASK_EXT;
            if (is_rgbd1 && !ExistsFile(mask_file1)) {
                // *.jpg.png => *.png
                CHECK(mask_file1.length() > strlen(MASK_EXT) + 5);
                mask_file1 = mask_file1.substr(0, mask_file1.length() - strlen(MASK_EXT) - 5) + "." + MASK_EXT;
            }
            if (ExistsFile(mask_file1)) {
                map_idx1_to_mask[i] = cv::imread(mask_file1, cv::IMREAD_UNCHANGED);
            }
        }

        if (map_idx1_to_mask.size() > 0) {
            std::cout << "Found " << map_idx1_to_mask.size() << " change masks" << std::endl;

            size_t total_filter_track = 0;
            size_t total_filter_point = 0;
            for (size_t i = 0; i < fused_points1.size(); i++) {
                const auto & vis1 = fused_points_vis1[i];
                auto & flags1 = view_discard_flag1[i];
                flags1.reserve(vis1.size());

                int filter_count = 0;
                Eigen::Vector3f xyz(fused_points1[i].x, fused_points1[i].y, fused_points1[i].z);
                for (size_t j = 0; j < vis1.size(); j++) {
                    uint32_t idx = vis1[j];
                    auto find = map_idx1_to_mask.find(idx);
                    if (find == map_idx1_to_mask.end()) continue;

                    auto & mask = find->second;
                    auto & image = model1.images[idx];
                    Eigen::Matrix3f R = Eigen::Map<const Eigen::RowMatrix3f>(image.GetR());
                    Eigen::Vector3f t = Eigen::Map<const Eigen::Vector3f>(image.GetT());
                    Eigen::Matrix3f intrinsic = Eigen::Map<const Eigen::RowMatrix3f>(image.GetK());
                    Eigen::Vector3f p = R * xyz + t;

                    unsigned char flag = 0;
                    if (p.z() > std::numeric_limits<float>::epsilon()) {
                        int x = intrinsic(0, 0) * p.x() / p.z() + intrinsic(0, 2) + 0.5f;
                        int y = intrinsic(1, 1) * p.y() / p.z() + intrinsic(1, 2) + 0.5f;
                        if (x >= 0 && y >= 0 && x < mask.cols && y < mask.rows) {
                            if (mask.at<uchar>(y, x)) {
                                flag = 255;
                                filter_count++;
                                total_filter_track++;
                            }
                        }
                    }

                    flags1.emplace_back(flag);
                }

                if ((int)vis1.size() - filter_count <= 1) {
                    all_discard_flag1[i] = 255;
                    total_filter_point++;
                }
            }
       
            std::cout << "Filtered " << total_filter_point << " points, " << total_filter_track << " tracks" << std::endl;
        }
    }

    for (size_t i = 0; i < fused_points1.size(); i++) {
        if (all_discard_flag1[i]) continue;

        fused_points3.emplace_back(fused_points1[i]);
    }
    fused_points3.insert(fused_points3.end(), fused_points2.begin(), fused_points2.end());
    WriteBinaryPlyPoints(fused_path3, fused_points3);

    if (has_sem) {
        WritePointsSemantic(fused_sem_path3, fused_points3);
        std::string fused_semvis_path3 = JoinPaths(GetParentDir(fused_path3), FUSION_SEM_NAME);
        WritePointsSemanticColor(fused_semvis_path3, fused_points3);
    }

    for (size_t i = 0; i < fused_points1.size(); i++) {
        if (all_discard_flag1[i]) continue;

        const auto & vis1 = fused_points_vis1[i];
        std::vector<uint32_t> vis3;
        vis3.reserve(vis1.size());
        for (size_t j = 0; j < vis1.size(); j++) {
            uint32_t v = vis1[j];
            if (view_discard_flag1[i].size() == 0 || !view_discard_flag1[i][j]) {
                vis3.emplace_back(map_idx1_to_idx3[v]);
            }
        }

        fused_points_vis3.emplace_back(vis3);
    }
    for (size_t i = 0; i < fused_points2.size(); i++) {
        std::vector<uint32_t> vis3;
        vis3.reserve(fused_points_vis2[i].size());
        for (auto & v : fused_points_vis2[i]) {
            vis3.emplace_back(map_idx2_to_idx3[v]);
        }
        fused_points_vis3.emplace_back(vis3);
    }
    CHECK_EQ(fused_points3.size(), fused_points_vis3.size());
    WritePointsVisibility(fused_vis_path3, fused_points_vis3);

    if (ExistsFile(fused_wgt_path1)) {
        CHECK(ExistsFile(fused_wgt_path2)) << "found " << fused_wgt_path1 << ", "
                                           << "but " << fused_wgt_path2 << " is not available. ";
        std::vector<std::vector<float> > points_weight1;
        std::vector<std::vector<float> > points_weight2;
        std::vector<std::vector<float> > points_weight3;
        ReadPointsWeight(fused_wgt_path1, points_weight1);
        ReadPointsWeight(fused_wgt_path2, points_weight2);
        for (size_t i = 0; i < points_weight1.size(); i++) {
            if (all_discard_flag1[i]) continue;

            const auto & wgt1 = points_weight1[i];
            std::vector<float> wgt3;
            wgt3.reserve(wgt1.size());
            for (size_t j = 0; j < wgt1.size(); j++) {
                float w = wgt1[j];
                if (view_discard_flag1[i].size() == 0 || !view_discard_flag1[i][j]) {
                    wgt3.emplace_back(w);
                }
            }

            points_weight3.emplace_back(wgt3);
            CHECK_EQ(fused_points_vis3[i].size(), points_weight3[i].size());
        }
        points_weight3.insert(points_weight3.end(), points_weight2.begin(), points_weight2.end());
        CHECK_EQ(fused_points3.size(), points_weight3.size());
        WritePointsWeight(fused_wgt_path3, points_weight3);
    }
}

int main(int argc, char ** argv)
{
	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

    std::string configuration_file_path1;
    std::string configuration_file_path2;
    std::string output_dense;
    if (argc == 3) {
        configuration_file_path1 = std::string(argv[1]);
        configuration_file_path2 = std::string(argv[2]);
    } else if (argc == 4) {
        configuration_file_path1 = std::string(argv[1]);
        configuration_file_path2 = std::string(argv[2]);
        output_dense = std::string(argv[3]);
    } else {
        std::cout << "Usage: " << argv[0] << std::endl;
        std::cout << "    <OLD_MVS_YAML> <NEW_MVS_YAML> [DENSE_OUTPUT]" << std::endl;
        return StateCode::NO_MATCHING_INPUT_PARAM;
    }

    Configurator param1;
    Configurator param2;
    param1.Load(configuration_file_path1.c_str());
    param2.Load(configuration_file_path2.c_str());

    std::string workspace_path1 = param1.GetArgument("workspace_path", "");
    std::string workspace_path2 = param2.GetArgument("workspace_path", "");
    std::string dense_path1 = JoinPaths(workspace_path1, "0", DENSE_DIR);
    std::string dense_path2 = JoinPaths(workspace_path2, "0", DENSE_DIR);
    std::string dense_path3 = output_dense.empty() ? JoinPaths(workspace_path2, "0", DENSE_DIR)
                                                   : output_dense;

    std::string image_type1 = param1.GetArgument("image_type", "perspective");
    std::string image_type2 = param2.GetArgument("image_type", "perspective");
    CHECK(image_type1 == "perspective" || image_type1 == "rgbd");
    CHECK(image_type2 == "perspective" || image_type2 == "rgbd");

    bool is_rgbd1 = image_type1 == "rgbd";
    bool is_rgbd2 = image_type2 == "rgbd";
    RGBDConversion conversion = RGBDConversion::None;
    if (is_rgbd1 && !is_rgbd2) {
        conversion = RGBDConversion::Perspective_To_RGBD;
    } else if (!is_rgbd1 && is_rgbd2) {
        conversion = RGBDConversion::RGBD_To_Perspective;
    }

    std::string image_path1 = JoinPaths(dense_path1, IMAGES_DIR);
    std::string image_path2 = JoinPaths(dense_path2, IMAGES_DIR);
    std::string image_path3 = JoinPaths(dense_path3, IMAGES_DIR);
    CHECK(ExistsDir(image_path1) && ExistsDir(image_path2));

    std::cout << "Merging " << image_path1 << " + " << image_path2 << " => " << image_path3 << std::endl;
    MergeImagePaths(image_path1, image_path2, image_path3, conversion);

    std::string image_orig_path1 = JoinPaths(dense_path1, IMAGES_ORIG_RES_DIR);
    std::string image_orig_path2 = JoinPaths(dense_path2, IMAGES_ORIG_RES_DIR);
    std::string image_orig_path3 = JoinPaths(dense_path3, IMAGES_ORIG_RES_DIR);
    if (ExistsDir(image_orig_path1)) {
        if (ExistsDir(image_orig_path2)) {
            std::cout << "Merging " << image_orig_path1 << " + " << image_orig_path2 << " => " << image_orig_path3 << std::endl;
            MergeImagePaths(image_orig_path1, image_orig_path2, image_orig_path3, conversion);
        } else {
            std::cout << "Merging " << image_orig_path1 << " + " << image_path2 << " => " << image_orig_path3 << std::endl;
            MergeImagePaths(image_orig_path1, image_path2, image_orig_path3, conversion);
        }
    }

    std::string semantic_path1 = JoinPaths(dense_path1, SEMANTICS_DIR);
    std::string semantic_path2 = JoinPaths(dense_path2, SEMANTICS_DIR);
    std::string semantic_path3 = JoinPaths(dense_path3, SEMANTICS_DIR);
    if (ExistsDir(semantic_path1)) {
        if (ExistsDir(semantic_path2)) {
            std::cout << "Merging " << semantic_path1 << " + " << semantic_path2 << " => " << semantic_path3 << std::endl;
        } else {
            std::cout << "Copying " << semantic_path1 << " => " << semantic_path3 << std::endl;
        }
        MergeImagePaths(semantic_path1, semantic_path2, semantic_path3, conversion);
    }

    std::string changemask_path1 = JoinPaths(dense_path1, CHANGE_MASKS_DIR);
    std::string changemask_path2 = JoinPaths(dense_path2, CHANGE_MASKS_DIR);
    std::string changemask_path3 = JoinPaths(dense_path3, CHANGE_MASKS_DIR);
    if (ExistsDir(changemask_path1)) {
        if (ExistsDir(changemask_path2)) {
            std::cout << "Merging " << changemask_path1 << " + " << changemask_path2 << " => " << changemask_path3 << std::endl;
        } else {
            std::cout << "Copying " << changemask_path1 << " => " << changemask_path3 << std::endl;
        }
        // Because chagne mask generator is shared for sparse/dense recons,
        // we don't change it's filename when RGBD for compatibility
        MergeImagePaths(changemask_path1, changemask_path2, changemask_path3, RGBDConversion::None);
    }

    mvs::Model model1, model2, model3;
    std::string sparse_path1 = JoinPaths(dense_path1, SPARSE_DIR);
    std::string sparse_path2 = JoinPaths(dense_path2, SPARSE_DIR);
    std::string sparse_path3 = JoinPaths(dense_path3, SPARSE_DIR);
    CHECK(ExistsDir(sparse_path1) && ExistsDir(sparse_path2));
    Reconstruction reconstruction1;
    Reconstruction reconstruction2;
    reconstruction1.ReadBinary(sparse_path1);
    reconstruction2.ReadBinary(sparse_path2);

    {
        mvs::Workspace::Options workspace_options;
        workspace_options.max_image_size = -1;
        workspace_options.image_path = image_path1;
        workspace_options.workspace_path = dense_path1;
        workspace_options.workspace_format = is_rgbd1 ? "rgbd" : "perspective";

        mvs::Workspace workspace(workspace_options);
        model1 = workspace.GetModel();
    }
    {
        mvs::Workspace::Options workspace_options;
        workspace_options.max_image_size = -1;
        workspace_options.image_path = image_path2;
        workspace_options.workspace_path = dense_path2;
        workspace_options.workspace_format = is_rgbd2 ? "rgbd" : "perspective";

        mvs::Workspace workspace(workspace_options);
        model2 = workspace.GetModel();
    }

    std::cout << "Merging " << sparse_path1 << " + " << sparse_path2 << " => " << sparse_path3 << std::endl;
    if (!boost::filesystem::exists(sparse_path3)) {
        boost::filesystem::create_directories(sparse_path3);
    }
    Reconstruction reconstruction3 = MergeReconstructions(reconstruction1, reconstruction2, conversion);
    reconstruction3.WriteBinary(sparse_path3);

    {
        mvs::Workspace::Options workspace_options;
        workspace_options.max_image_size = -1;
        workspace_options.image_path = image_path3;
        workspace_options.workspace_path = dense_path3;
        workspace_options.workspace_format = is_rgbd1 ? "rgbd" : "perspective";

        mvs::Workspace workspace(workspace_options);
        model3 = workspace.GetModel();
    }

    std::string sparse_orig_path1 = JoinPaths(dense_path1, SPARSE_ORIG_RES_DIR);
    std::string sparse_orig_path2 = JoinPaths(dense_path2, SPARSE_ORIG_RES_DIR);
    std::string sparse_orig_path3 = JoinPaths(dense_path3, SPARSE_ORIG_RES_DIR);
    if (ExistsDir(sparse_orig_path1)) {
        Reconstruction reconstruction1;
        reconstruction1.ReadBinary(sparse_orig_path1);

        Reconstruction reconstruction2;
        if (ExistsDir(sparse_orig_path2)) {
            std::cout << "Merging " << sparse_orig_path1 << " + " << sparse_orig_path2 << " => " << sparse_orig_path3 << std::endl;
            reconstruction2.ReadBinary(sparse_orig_path2);
        } else {
            std::cout << "Merging " << sparse_orig_path1 << " + " << sparse_path2 << " => " << sparse_orig_path3 << std::endl;
            reconstruction2.ReadBinary(sparse_path2);
        }
        if (!boost::filesystem::exists(sparse_orig_path3)) {
            boost::filesystem::create_directories(sparse_orig_path3);
        }

        Reconstruction reconstruction3;
        reconstruction3 = MergeReconstructions(reconstruction1, reconstruction2, conversion);
        reconstruction3.WriteBinary(sparse_orig_path3);
    }

    std::string fused_path1 = JoinPaths(dense_path1, FUSION_NAME);
    std::string fused_path2 = JoinPaths(dense_path2, FUSION_NAME);
    std::string fused_path3 = JoinPaths(dense_path3, FUSION_NAME);
    if (ExistsFile(fused_path1) && ExistsFile(fused_path2)) {
        std::cout << "Merging " << fused_path1 << " + " << fused_path2 << " => " << fused_path3 << std::endl;
        MergeFusedPlys(dense_path1, model1, dense_path2, model2, dense_path3, model3, is_rgbd1);
    }

    return 0;
}