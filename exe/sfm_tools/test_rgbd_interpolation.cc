//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "util/misc.h"
#include "util/imageconvert.h"
#include "base/undistortion.h"
#include "base/reconstruction_manager.h"
#include "base/essential_matrix.h"
#include "base/homography_matrix.h"
#include "controllers/patch_match_controller.h"
#include "optim/ransac/loransac.h"
#include "estimators/essential_matrix.h"
#include "estimators/similarity_transform.h"

#include "base/version.h"
#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "base/image.h"
#include <opencv2/core/eigen.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include "util/misc.h"
using namespace boost::multiprecision;

std::string configuration_file_path;
using namespace sensemap;
using namespace sensemap::mvs;
double fps = 30;
double align_max_reproj_error = 30.0;
struct RGBDInfo
{
    std::string calib_cam;
    std::string sub_path;
    std::string rgbd_camera_params;
    Eigen::Matrix3d extra_R;// extrinsic for pro2 or one-r
    Eigen::Vector3d extra_T;
    int timestamp = -1;//whether file format is timestamp(1) or not(0) or auto detect(-1)

    bool has_force_offset = false;
    int force_offset = 0;
};
std::vector<RGBDInfo> rgbd_infos;
std::string target_subpath;

struct PixelError{
    double pixel_error;
    int cnt;
    int offset;
    PixelError(double pixel_error_,int cnt_, int offset_)
    {
        pixel_error = pixel_error_;
        cnt = cnt_;
        offset = offset_;
    }
    bool operator < (const PixelError &a) const
    {
        return a.cnt<cnt;
    }
};

void DeleteRedundantImages(Reconstruction* reconstruction_,std::string image_path)
{
    auto name_to_index = [](const std::string & s) {
        return std::atoll(boost::filesystem::path(s).stem().string().c_str());
    };

    std::unordered_set<std::string> filenames;
    {
        std::vector<std::string> files = GetRecursiveFileList(JoinPaths(image_path, target_subpath));
        for (auto & file : files) {
            filenames.insert(boost::filesystem::path(file).filename().string());
        }
    }

    if (filenames.size() >= 2) {
        std::vector<std::string> files(filenames.begin(), filenames.end());
        std::sort(files.begin(), files.end(), [&](const std::string & a, const std::string & b){
            return name_to_index(a) < name_to_index(b);
        });

        int64_t file_index_1 = name_to_index(files[files.size() - 2]);
        int64_t file_index_2 = name_to_index(files[files.size() - 1]);
        int64_t interval = file_index_2 - file_index_1;
        std::cout << "Tailing frame interval " << interval << std::endl;

        int64_t file_index_3 = name_to_index(files[0]);
        int64_t file_index_4 = name_to_index(files[1]);
        int64_t interval2 = file_index_4 - file_index_3;
        std::cout << "Starting frame interval " << interval2 << std::endl;

        const auto & map_image_names0 = reconstruction_->GetImageNames();
        std::unordered_map<std::string, image_t> map_image_names;
        for (auto & item : map_image_names0) {
            map_image_names.emplace(boost::filesystem::path(item.first).filename().string(), item.second);
        }

        for (int64_t i = files.size() - 2; i >= 0; i--) {
            int64_t file_index_target = name_to_index(files[i + 1]);
            for ( ; i >= 0; i--) {
                uint64_t diff = file_index_target - name_to_index(files[i]);
                if (interval2 <= 3) {
                    if (diff == 0 || diff >= interval) break;
                } else {
                    // avoid rounding issues
                    if (diff == 0 || diff >= interval - 1) break;
                }

                auto find = map_image_names.find(files[i]);
                if (find != map_image_names.end()) {
                    reconstruction_->DeleteImage(find->second);
                }
            }
        }
    }
}

void OptimizeDeltaTime(Reconstruction* reconstruction_,std::string image_path,std::string result_path)
{
    std::cout<<"Align By Reprojection Error"<<std::endl;

    std::map<int64_t, image_t> map_index_to_image_id;
    for (auto & item : reconstruction_->GetImageNames()) {
        if (IsInsideSubpath(item.first, target_subpath)) {
            int64_t index = std::atoll(boost::filesystem::path(item.first).stem().string().c_str());
            CHECK(map_index_to_image_id.count(index) == 0) << "Duplicated image numeric name " << item.first;
            
            map_index_to_image_id[index] = item.second;
        }
    }

    int64_t target_ms_per_frame = 1;
    std::map<int64_t, int> ms_per_frame_count;
    for (auto iter1 = map_index_to_image_id.begin(); iter1 != map_index_to_image_id.end(); iter1++) {
        auto iter2 = iter1;
        iter2++;
        if (iter2 != map_index_to_image_id.end()) {
            int64_t diff = iter2->first - iter1->first;
            if (ms_per_frame_count.count(diff)) {
                ms_per_frame_count[diff] += 1;
            } else {
                ms_per_frame_count[diff] = 1;
            }
        }
    }
    for (auto item : ms_per_frame_count) {
        if (item.second < 5) continue;

        target_ms_per_frame = item.first;
        break;
    }

    for(int i_cam = 0;i_cam<rgbd_infos.size();i_cam++)
    {
        std::string sub_path = rgbd_infos[i_cam].sub_path;
        std::string camth = rgbd_infos[i_cam].calib_cam;
        int local_cam_id = camth.empty() ? -1 : std::atoi(camth.substr(3, camth.size()).c_str());
        std::cout << "local_cam_id:" << local_cam_id<<std::endl;
        Eigen::Matrix3d extra_R = rgbd_infos[i_cam].extra_R;
        Eigen::Vector3d extra_T = rgbd_infos[i_cam].extra_T;

        double dis_c = (-extra_R.inverse() * extra_T).norm();
        std::cout << "dis_c:" << dis_c << std::endl;

        std::string rgbd_path = JoinPaths(image_path, sub_path);
        std::vector<std::string> rgbd_files = GetRecursiveFileList(rgbd_path);
        std::vector<std::string> rgbd_names;
        for (const auto & str : rgbd_files) {
            rgbd_names.emplace_back(GetRelativePath(image_path, str));
        }
        std::sort(rgbd_names.begin(), rgbd_names.end(), 
        [] (const std::string & a, const std::string & b)
        {
            if (a.size() < b.size())
            {
                return true;
            }
            else if (a.size() > b.size())
            {
                return false;
            }
            else
            {
                return a < b;
            }
        });

        const auto register_image_ids = reconstruction_->RegisterImageIds();

        int64_t rgbd_timestamp_start = std::atoll(boost::filesystem::path(rgbd_names[0]).stem().string().c_str());
        int64_t target_timestamp_start = map_index_to_image_id.begin()->first;
        int64_t timstamp_diff = target_timestamp_start - rgbd_timestamp_start;
        bool timestamp = false;
        if (rgbd_infos[i_cam].timestamp < 0) {
            if (rgbd_timestamp_start > 10000000 || target_timestamp_start > 10000000) {
                timestamp = true;
            }
        } else if (rgbd_infos[i_cam].timestamp > 0) {
            timestamp = true;
        }
        if (timestamp) {
            std::cout << "Start timestamp(Target): " << target_timestamp_start << std::endl;
            std::cout << "Start timestamp(RGBD): " << rgbd_timestamp_start << std::endl;
            std::cout << "Estimated " << target_ms_per_frame << " ms per frame" << std::endl;
        }

        int64_t offset = 0;
        double pixel_error = -1.0f;
        if (!rgbd_infos[i_cam].has_force_offset)
        {
            std::map<int64_t, std::pair<int, double>> map_offset_count_score;
            #pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < register_image_ids.size(); i++) {
                const auto & rgbd_image = reconstruction_->Image(register_image_ids[i]);
                const auto & rgbd_camera = reconstruction_->Camera(rgbd_image.CameraId());
                const auto rgbd_C = -QuaternionToRotationMatrix(rgbd_image.Qvec()).transpose()*rgbd_image.Tvec();
                const int64_t rgbd_index = std::atoll(boost::filesystem::path(rgbd_image.Name()).stem().string().c_str());
                if (!IsInsideSubpath(rgbd_image.Name(), sub_path)) continue;

                for (const auto & point2d : rgbd_image.Points2D()) 
                {
                    if (!point2d.HasMapPoint()) continue;
                    auto map_pt = reconstruction_->MapPoint(point2d.MapPointId());

                    for (const auto & track : map_pt.Track().Elements()) 
                    {
                        const auto & target_image = reconstruction_->Image(track.image_id);
                        const int64_t target_index = std::atoll(boost::filesystem::path(target_image.Name()).stem().string().c_str());
                        if (!IsInsideSubpath(target_image.Name(), target_subpath)) continue;
                        if (timestamp) {
                            if (std::abs(target_index - rgbd_index - timstamp_diff) > 30000) continue;
                        } else {
                            if (std::abs(target_index - rgbd_index) > 1000) continue;
                        }
                        const auto target_C = -QuaternionToRotationMatrix(target_image.Qvec()).transpose()*target_image.Tvec();
                        auto dis = (rgbd_C - target_C).norm();
                        if (dis < dis_c * 0.5 - 0.2 || dis > dis_c * 2.0 + 0.2) continue;

                        auto target_Qvec = target_image.Qvec();
                        auto target_Tvec = target_image.Tvec();
                        Eigen::Quaterniond target_Qvec_eigen(target_Qvec(0), target_Qvec(1), target_Qvec(2), target_Qvec(3));
                        Eigen::Matrix3d target_R = target_Qvec_eigen.toRotationMatrix();

                        if (local_cam_id > 0) {
                            Eigen::Vector4d local_qvec;
                            Eigen::Vector3d local_tvec;
                            const auto target_camera = reconstruction_->Camera(target_image.CameraId());
                            target_camera.GetLocalCameraExtrinsic(local_cam_id, local_qvec, local_tvec);
                            target_R = QuaternionToRotationMatrix(local_qvec) * target_R;
                            target_Tvec = QuaternionToRotationMatrix(local_qvec) * target_Tvec + local_tvec;
                        }

                        Eigen::Vector3d target_cam_coord  = target_R * map_pt.XYZ() + target_Tvec;
                        Eigen::Vector3d estimated_cam_coord = extra_R.inverse() * (target_cam_coord - extra_T);
                        auto estimated_img_coord = rgbd_camera.WorldToImage(estimated_cam_coord.hnormalized());
                        double error2 = (estimated_img_coord - point2d.XY()).squaredNorm();
                        if (error2 < align_max_reproj_error * align_max_reproj_error)
                        {
                            int64_t offset = target_index - rgbd_index;
                            #pragma omp critical
                            {
                                auto find = map_offset_count_score.find(offset);
                                if (find == map_offset_count_score.end()) {
                                    map_offset_count_score[offset] = std::make_pair(1, error2);
                                } else {
                                    find->second.first += 1;
                                    find->second.second += error2;
                                }
                            }
                            // cur_count++;
                            // cur_energy += (u - x) * (u - x) + (v - y) * (v - y);
                        }
                    }
                }
            }

            if (timestamp) {
                // merge neighboring timestamps
                int64_t diff_thresh = target_ms_per_frame / 4;
                if (diff_thresh > 0) {
                    std::cout << "Merging neighboring timestamps using thresh " << diff_thresh << " ms" << std::endl;

                    std::map<int64_t, std::pair<int, double>> map_offset_count_score2;
                    for (auto iter1 = map_offset_count_score.begin(); iter1 != map_offset_count_score.end(); iter1++) {
                        auto item = iter1->second;
                        for (auto iter2 = iter1; iter2 != map_offset_count_score.end(); iter2++) {
                            if (iter2 != iter1) {
                                if (iter2->first - iter1->first <= diff_thresh) {
                                    item.first += iter2->second.first;
                                    item.second += iter2->second.second;
                                } else {
                                    break;
                                }
                            }
                        }
                        for (auto iter2 = iter1; iter2 != map_offset_count_score.end(); iter2--) {
                            if (iter2 != iter1) {
                                if (iter1->first - iter2->first <= diff_thresh) {
                                    item.first += iter2->second.first;
                                    item.second += iter2->second.second;
                                } else {
                                    break;
                                }
                            }
                        }

                        map_offset_count_score2[iter1->first] = item;
                    }
                    std::swap(map_offset_count_score, map_offset_count_score2);
                }
            }

            // Remove offsets with low counts
            int max_count = 0;
            for (auto item : map_offset_count_score) {
                max_count = std::max(item.second.first, max_count);
            }
            std::set<int64_t> offsets_to_del;
            for (auto item : map_offset_count_score) {
                if (item.second.first < max_count * 0.5 || item.second.first < 200) {
                    offsets_to_del.emplace(item.first);
                }
            }
            for (auto offset : offsets_to_del) {
                map_offset_count_score.erase(offset);
            }

            // Pick offset with smallest error
            if (map_offset_count_score.size() > 0) {
                std::cout << "Candidates: " << std::endl;
                pixel_error = std::numeric_limits<double>::max();
                for (auto item : map_offset_count_score) {
                    double error = std::sqrt(item.second.second / item.second.first);
                    std::cout << item.first << " " << item.second.first << " " << error << std::endl;
                    if (std::sqrt(item.second.second / item.second.first) < pixel_error) {
                        offset = item.first;
                        pixel_error = error;
                    }
                }
            } else {
                std::cout<<"ERROR: i_cam:" << i_cam << " failed!"<<std::endl;
                continue;
            }
        }
        else
        {
            //test
            offset = rgbd_infos[i_cam].force_offset;
            // end test
        }

        std::cout << "offset: " << offset << (timestamp ? " ms" : "") <<std::endl;
        std::cout << "pixel error: " << pixel_error << std::endl;

        // first delete old rgbd images
        image_t max_image_id = -1;
        camera_t sub_path_camera = 1;
        for(int i = 0; i < register_image_ids.size(); i++)
        {
            if (max_image_id == -1)
            {
                max_image_id = register_image_ids[i];
            }
            else
            {
                max_image_id = std::max(max_image_id, register_image_ids[i]);
            }

            const auto &image = reconstruction_->Image(register_image_ids[i]);
            if (IsInsideSubpath(image.Name(), sub_path))
            {
                sub_path_camera = image.CameraId();
                reconstruction_->DeleteImage(register_image_ids[i]);
            }
        }

        for (int i = 0; i < rgbd_names.size(); i++)
        {
            image_t target_image_id = -1;
            if (!timestamp)
            {
                int64_t index = std::atoll(boost::filesystem::path(rgbd_names[i]).stem().string().c_str());
                int64_t target_index = index + offset;
                if (map_index_to_image_id.count(target_index) == 0) continue;
                target_image_id = map_index_to_image_id[target_index];
            }
            else
            {
                int64_t timestamp = std::atoll(boost::filesystem::path(rgbd_names[i]).stem().string().c_str());
                int64_t target_timestamp = timestamp + offset;
                int64_t diff_thresh = (target_ms_per_frame + 1) / 2;
                for (int64_t diff = 0; diff < diff_thresh; diff += 1) {
                    if (map_index_to_image_id.count(target_timestamp + diff)) {
                        target_image_id = map_index_to_image_id[target_timestamp + diff];
                        break;
                    } else if (map_index_to_image_id.count(target_timestamp - diff)) {
                        target_image_id = map_index_to_image_id[target_timestamp - diff];
                        break;
                    }
                }
            }
            if (target_image_id == -1 || !reconstruction_->IsImageRegistered(target_image_id)) continue;
            
            const auto & target_image = reconstruction_->Image(target_image_id);
            auto target_qvec = target_image.Qvec();
            auto target_tvec = target_image.Tvec();
            std::cout << "Register " << rgbd_names[i] << " -> " << target_image.Name() << std::endl;

            Eigen::Quaterniond target_q(target_qvec(0),target_qvec(1),target_qvec(2),target_qvec(3));
            target_q.normalize();
            Eigen::Matrix3d target_R = target_q.toRotationMatrix();
            if (local_cam_id > 0)
            {
                Eigen::Vector4d local_qvec;
                Eigen::Vector3d local_tvec;
                const auto & target_camera = reconstruction_->Camera(target_image.CameraId());
                target_camera.GetLocalCameraExtrinsic(local_cam_id, local_qvec, local_tvec);
                target_R = QuaternionToRotationMatrix(local_qvec) * target_R;
                target_tvec = QuaternionToRotationMatrix(local_qvec) * target_tvec + local_tvec;
            }

            Eigen::Matrix3d RR = extra_R.inverse() * target_R;
            Eigen::Vector3d tt = extra_R.inverse() * (target_tvec - extra_T);

            sensemap::Image image;
            image.SetImageId(++max_image_id);

            image.SetQvec(RotationMatrixToQuaternion(RR));
            image.NormalizeQvec();
            image.SetTvec(tt);

            image.SetCameraId(sub_path_camera);
            auto camera = reconstruction_->Camera(sub_path_camera);
            image.SetUp(camera);

            image.Name() = rgbd_names[i];
            reconstruction_->AddImage(image);
            reconstruction_->RegisterImage(image.ImageId());
        }
        std::cout<<"i_cam:"<<i_cam<<" finished!"<<std::endl;
    }

    DeleteRedundantImages(reconstruction_, image_path);
    std::cout << "Registered: " << reconstruction_->RegisterImageIds().size() << std::endl;
    reconstruction_->WriteReconstruction(result_path);
}

void RescaleByStatitics(Reconstruction* reconstruction_,std::string image_path,std::string result_path)
{
    auto image_names = reconstruction_->GetImageNames();
    std::vector<std::string> rgbd_names;
    for (auto & image_name : image_names) {
        if (!IsFileRGBD(image_name.first)) continue;
        rgbd_names.emplace_back(image_name.first);
    }
    const int step = std::max(1.0, rgbd_names.size() * 0.007);

    double scale = 0;
    std::vector<double> scale_candidates;
    #pragma omp parallel
    {
        std::vector<double> _scale_candidates;

        #pragma omp for schedule(dynamic, 1)
        for(int i = 0; i < rgbd_names.size(); i += step)
        {
            const std::string & image_name = rgbd_names[i];
            
            const image_t image_id = image_names[image_name];
            if ((!reconstruction_->ExistsImage(image_id))) continue;

            const auto & image = reconstruction_->Image(image_id);
            const auto & camera = reconstruction_->Camera(image.CameraId());
            if (!image.IsRegistered()) continue;

            const int width = camera.Width();
            const int height = camera.Height();
            RGBDData data;
            ExtractRGBDData(JoinPaths(image_path, image_name), RGBDReadOption::NoColor(), data);
            if (!data.HasRGBDCalibration()) continue;

            MatXf warped_depthmap(width, height, 1);
            UniversalWarpDepthMap(warped_depthmap, data.depth, data.color_camera, data.depth_camera, data.depth_RT.cast<float>());

            auto image_qvec = image.Qvec();
            auto image_tvec = image.Tvec();
            Eigen::Quaterniond image_q(image_qvec(0), image_qvec(1), image_qvec(2), image_qvec(3));
            auto R = image_q.toRotationMatrix();

            for (const auto & point2d : image.Points2D())
            {
                if (!point2d.HasMapPoint()) continue;
                auto mappoint_id = point2d.MapPointId();

                auto pt_3d = reconstruction_->MapPoint(mappoint_id).XYZ();
                auto pt_cam = R * pt_3d + image_tvec;
                if (pt_cam.z() <= 0) continue;
                
                double pt_depth = pt_cam.z();
                auto image_coord = data.color_camera.WorldToImage(pt_cam.hnormalized());

                int nx = image_coord(0) + 0.5f;
                int ny = image_coord(1) + 0.5f;
                if (nx < 0 || nx >= warped_depthmap.GetWidth() ||
                    ny < 0 || ny >= warped_depthmap.GetHeight()
                ) {
                    continue;    
                }

                double rgbd_depth = warped_depthmap.Get(ny, nx);
                if (rgbd_depth > 0.01) {
                    _scale_candidates.push_back(pt_cam.z() / rgbd_depth);
                }
            }
        }
    
        #pragma omp critical
        {
            scale_candidates.reserve(scale_candidates.size() + _scale_candidates.size());
            scale_candidates.insert(scale_candidates.end(), _scale_candidates.begin(), _scale_candidates.end());
        }
    }

    if (scale_candidates.empty()) {
        std::cout << "ERROR: no candidates for scale adjustment!" << std::endl;
        std::cout << "ERROR: no candidates for scale adjustment!" << std::endl;
        std::cout << "ERROR: no candidates for scale adjustment!" << std::endl;
        return;
    }

    double threshold = 0.02f;
    //Ransac
    int cc_count = 0;
    std::cout << "scale candidate size:" << scale_candidates.size() << std::endl;
    #pragma omp parallel for schedule(dynamic, 16)
    for(int i = 0; i < scale_candidates.size(); i++)
    {
        int cc = 0;
        for(int j = 0; j < scale_candidates.size(); j++)
        {
            if (j==i)
            {
                continue;
            }
            if (std::fabs(scale_candidates[i] - scale_candidates[j]) / std::fabs(scale_candidates[i]) < threshold)
            {
                cc++;
            }

        }
        if(cc > cc_count)
        {
            #pragma omp critical
            {
                // dual check is necessary
                if(cc > cc_count)
                {
                    cc_count = cc;
                    scale = scale_candidates[i];
                }
            }
        }
    }
    printf("The %.2f%% scales in inliner\n",(double)cc_count/scale_candidates.size()*100.f);
    //end Ransac
    auto mappoints = reconstruction_->MapPoints();
    std::cout<<"scale:"<<1.f/scale<<std::endl;
    if(scale)
    {
        reconstruction_->Rescale(1.f/scale);
        for (auto & pair : reconstruction_->Cameras()) {
            auto & camera = reconstruction_->Camera(pair.first);
            for (auto & tvec : camera.LocalTvecs()) {
                tvec *= 1.f/scale;
            }
        }
        reconstruction_->WriteReconstruction(result_path);
    }
}

int main(int argc, char *argv[]) {
    PrintHeading("SenseMap.  Copyright(c) 2021, SenseTime Group.");
    PrintHeading(std::string("Version: sfm-rgbd-interpolation-")+__VERSION__);

	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

	std::string image_path = param.GetArgument("image_path", "");
	std::string workspace_path = param.GetArgument("workspace_path", "");
    std::string camera_param_file = param.GetArgument("camera_param_file","");
    std::string rgbd_params_file;
    if (camera_param_file == "") return 0;

    bool rescale_flag = param.GetArgument("rescale",0);
    bool rescale_align_flag = param.GetArgument("rescale_align",1);
    std::cout<<"rescale_flag:"<<rescale_flag<<std::endl;
    std::cout<<"rescale_align_flag:"<<rescale_align_flag<<std::endl;
    YAML::Node node = YAML::LoadFile(camera_param_file);
    int num_cameras = param.GetArgument("num_cameras",0);
    int num_to_register = num_cameras - 1;
    std::cout<<"num to register:"<<num_to_register<<std::endl;
    if (num_to_register <= 0) return 0;

    target_subpath = node["sub_path_0"].as<std::string>();
    for(int i = 1; i <= num_to_register; i++)
    {
        RGBDInfo tmp;
        tmp.calib_cam = node["calib_cam_"+std::to_string(i)].as<std::string>();
        tmp.sub_path = node["sub_path_"+std::to_string(i)].as<std::string>();
        YAML::Node cv_mat_node = node["extrinsic_" + std::to_string(i)];
        if (cv_mat_node.IsDefined())
        {
            std::vector<double> mat_data = cv_mat_node["data"].as<std::vector<double> >();
            int mat_rows = cv_mat_node["rows"].as<int>();
            int mat_cols = cv_mat_node["cols"].as<int>();
            CHECK_EQ(mat_data.size(), mat_rows * mat_cols);
            cv::Mat1d extrinsic(mat_rows, mat_cols, mat_data.data());

            cv::Mat extra_R_mat, extra_T_mat;
            extrinsic(cv::Rect(0, 0, 3, 3)).copyTo(extra_R_mat);
            cv::cv2eigen(extra_R_mat, tmp.extra_R);
            extrinsic(cv::Rect(3, 0, 1, 3)).copyTo(extra_T_mat);
            cv::cv2eigen(extra_T_mat, tmp.extra_T);
            tmp.extra_T /= 1000.f;
        }
        
        if(node["timestamp_"+std::to_string(i)].IsDefined())
        {
            tmp.timestamp = node["timestamp_"+std::to_string(i)].as<int>();
        }

        // if(!fs["rgbd_params_file_"+std::to_string(i)].empty())
        if(node["rgbd_params_file_"+std::to_string(i)].IsDefined())
        {
            Eigen::Matrix3f rgb_K, depth_K;
            Eigen::Matrix4f RT;
            // fs["rgbd_params_file_"+std::to_string(i)]>>rgbd_params_file;
            rgbd_params_file = node["rgbd_params_file_"+std::to_string(i)].as<std::string>();
            std::cout<<rgbd_params_file<<std::endl;
            auto calib_reader = GetCalibBinReaderFromName(rgbd_params_file);
            calib_reader->ReadCalib(rgbd_params_file);

            tmp.rgbd_camera_params = calib_reader->ToParamString();
        }

        if (i + 1 < argc) {
            int offset;
            std::stringstream ss(argv[i + 1]);
            if (ss >> offset) {
                tmp.has_force_offset = true;
                tmp.force_offset = offset;
            }
        }

        std::cout<<tmp.extra_R<<std::endl;
        std::cout<<tmp.extra_T<<std::endl;
        rgbd_infos.push_back(tmp);
    }

    //load the sfm result
    std::string input_workspace_path_0 = workspace_path+"/0";
    std::string output_workspace_path_0 = workspace_path+"/0";
    Reconstruction* reconstruction_ = new Reconstruction();
    reconstruction_->ReadReconstruction(input_workspace_path_0,1);
    if(rescale_flag)
    {
        RescaleByStatitics(reconstruction_,image_path,output_workspace_path_0);
    }
    if(rescale_align_flag)
    {
        OptimizeDeltaTime(reconstruction_,image_path,output_workspace_path_0);
    }

	return 0;
}
