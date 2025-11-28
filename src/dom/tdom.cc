// #define POISSON_BLENDING

#include <queue>
#include <sstream>
#include <iterator>
#include <ogr_spatialref.h>

#include "base/common.h"
#include "base/cost_functions.h"
#include "util/misc.h"
#include "util/bitmap.h"
#include "util/math.h"
#include "util/threading.h"
#include "util/timer.h"
#include "util/proc.h"
#include "util/gps_reader.h"
#include "util/threading.h"
#include "util/histogram.h"
#include "util/exception_handler.h"
#include "graph/minimum_spanning_tree.h"

#include "tdom.h"
#ifdef POISSON_BLENDING
#include "poisson_blending.h"
#endif

#include "mapmap/full.h"

#ifdef WITH_GCO_OPTIMIZER
#include "GCO/GCoptimization.h"
#include "FullGCHelper.h"
#endif

#define MAX_DIST 1000000
#define MAX_PRECISION std::setprecision(std::numeric_limits<double>::digits10 + 1)
#define GCO_MAX_DATATERM 500
#define GCO_MAX_SMOOTHTERM 6000
#define MAX_CAPACITY 50

const int dirs[8][2] = { {-1, 0}, {0, 1}, {1, 0}, {0, -1},
                         {-1, -1}, {-1, 1}, {1, 1}, {1, -1} };

namespace sensemap {
namespace tdom {

float ImageQuality(const cv::Mat& bitmap) {
    cv::Mat dst;
    cv::Laplacian(bitmap, dst, CV_32FC1, 3);

    cv::Mat mean, stddev;
    cv::meanStdDev(dst, mean, stddev);

    return stddev.at<double>(0, 0);
}

void PrintSolverSummary(const ceres::Solver::Summary &summary) {
    std::cout << std::right << std::setw(16) << "Residuals : ";
    std::cout << std::left << summary.num_residuals_reduced << std::endl;

    std::cout << std::right << std::setw(16) << "Parameters : ";
    std::cout << std::left << summary.num_effective_parameters_reduced
              << std::endl;

    std::cout << std::right << std::setw(16) << "Iterations : ";
    std::cout << std::left
              << summary.num_successful_steps + summary.num_unsuccessful_steps
              << std::endl;

    std::cout << std::right << std::setw(16) << "Time : ";
    std::cout << std::left << summary.total_time_in_seconds << " [s]"
              << std::endl;

    std::cout << std::right << std::setw(16) << "Initial cost : ";
    std::cout << std::right << std::setprecision(6)
              << std::sqrt(summary.initial_cost / summary.num_residuals_reduced)
              << " [px]" << std::endl;

    std::cout << std::right << std::setw(16) << "Final cost : ";
    std::cout << std::right << std::setprecision(6)
              << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
              << " [px]" << std::endl;

    std::cout << std::right << std::setw(16) << "Termination : ";

    std::string termination = "";

    switch (summary.termination_type) {
        case ceres::CONVERGENCE:
            termination = "Convergence";
            break;
        case ceres::NO_CONVERGENCE:
            termination = "No convergence";
            break;
        case ceres::FAILURE:
            termination = "Failure";
            break;
        case ceres::USER_SUCCESS:
            termination = "User success";
            break;
        case ceres::USER_FAILURE:
            termination = "User failure";
            break;
        default:
            termination = "Unknown";
            break;
    }

    std::cout << std::right << termination << std::endl;
    std::cout << std::endl;
}

void SplitSpace(std::vector<Eigen::Vector2i> proj_coords, const int num_image_per_block, 
                const int axis, const Eigen::Vector4i & range,
                std::vector<Eigen::Vector4i> & blocks, int no) {
    if (proj_coords.size() <= num_image_per_block) {
        blocks.push_back(range);
        return;
    }
    if (axis == 0) {
        // std::cout << StringPrintf("split X (level = %d)", no) << std::endl;
        // std::cout << range.transpose() << std::endl;
        // std::cout << proj_coords.size() << std::endl;
        int nth = proj_coords.size() / 2;
        std::nth_element(proj_coords.begin(), proj_coords.begin() + nth, proj_coords.end(),
            [&](const Eigen::Vector2i & a, const Eigen::Vector2i & b) {
                return a.x() < b.x();
            });
        int pivot = proj_coords.at(nth).x();
        std::vector<Eigen::Vector2i> left(proj_coords.begin(), proj_coords.begin() + nth);
        std::vector<Eigen::Vector2i> right(proj_coords.begin() + nth, proj_coords.end());
        Eigen::Vector4i left_range = range;
        left_range[2] = pivot;
        Eigen::Vector4i right_range = range;
        right_range[0] = pivot;
        // std::cout << "left: " << left_range.transpose() << std::endl;
        // std::cout << "right: " << right_range.transpose() << std::endl;
        int next_axis = (left_range[2] - left_range[0]) > (left_range[3] - left_range[1]) ? 0 : 1;
        SplitSpace(left, num_image_per_block, next_axis, left_range, blocks, no + 1);
        next_axis = (right_range[2] - right_range[0]) > (right_range[3] - right_range[1]) ? 0 : 1;
        SplitSpace(right, num_image_per_block, next_axis, right_range, blocks, no + 1);
    } else if (axis == 1) {
        // std::cout << StringPrintf("split Y (level = %d)", no) << std::endl;
        // std::cout << range.transpose() << std::endl;
        // std::cout << proj_coords.size() << std::endl;
        int nth = proj_coords.size() / 2;
        std::nth_element(proj_coords.begin(), proj_coords.begin() + nth, proj_coords.end(),
            [&](const Eigen::Vector2i & a, const Eigen::Vector2i & b) {
                return a.y() < b.y();
            });
        int pivot = proj_coords.at(nth).y();
        std::vector<Eigen::Vector2i> top(proj_coords.begin(), proj_coords.begin() + nth);
        std::vector<Eigen::Vector2i> bottom(proj_coords.begin() + nth, proj_coords.end());
        Eigen::Vector4i top_range = range;
        top_range[3] = pivot;
        Eigen::Vector4i bottom_range = range;
        bottom_range[1] = pivot;
        // std::cout << "top: " << top_range.transpose() << std::endl;
        // std::cout << "bottom: " << bottom_range.transpose() << std::endl;
        int next_axis = (top_range[2] - top_range[0]) > (top_range[3] - top_range[1]) ? 0 : 1;
        SplitSpace(top, num_image_per_block, next_axis, top_range, blocks, no + 1);
        next_axis = (bottom_range[2] - bottom_range[0]) > (bottom_range[3] - bottom_range[1]) ? 0 : 1;
        SplitSpace(bottom, num_image_per_block, next_axis, bottom_range, blocks, no + 1);
    }
}

// #include "utm_zone.imp"

TDOM::TDOM(const TDOMOptions &options, const std::string workspace_path)
    : options_(options),
      workspace_path_(workspace_path),
      num_reconstruction_(0) {}

void TDOM::Run() {
    ReadWorkspace();

    const float in_gsd = options_.gsd;
    const float max_oblique_angle = options_.max_oblique_angle;

    for (int rec_idx = 0; rec_idx < num_reconstruction_; ++rec_idx) {
        Timer timer;
        timer.Start();

        PrintHeading1(StringPrintf("Process Reconstruction %d", rec_idx));

        auto reconstruction_path = JoinPaths(workspace_path_, std::to_string(rec_idx));
        auto dense_reconstruction_path = JoinPaths(reconstruction_path, DENSE_DIR);

        std::ifstream fin(JoinPaths(workspace_path_, "ned_to_ecef.txt"), std::ifstream::in);
        if (fin.is_open()) {
            fin >> trans_(0, 0) >> trans_(0, 1) >> trans_(0, 2) >> trans_(0, 3);
            fin >> trans_(1, 0) >> trans_(1, 1) >> trans_(1, 2) >> trans_(1, 3);
            fin >> trans_(2, 0) >> trans_(2, 1) >> trans_(2, 2) >> trans_(2, 3);
        }
        fin.close();
        std::cout << "ned to ecef: " << std::endl;
        std::cout << trans_ << std::endl;

        bool color_harmonization = options_.color_harmonization;
        std::cout << "color harmonization: " << color_harmonization << std::endl;

        // Load Model.
        SetUp(rec_idx);

        ComputeTransformationToUTM();

        // Estimate Ground Sample Distance.
        EstimateGSD();

        valid_images_.clear();
        valid_images_.resize(model_.images.size(), 0);

        const float oblique_thres = std::cos(DEG2RAD(max_oblique_angle));
        int num_image_oblique = 0;
        for (int i = 0; i < model_.images.size(); ++i) {
            const mvs::Image & image = model_.images.at(i);
            Eigen::Vector3f ray(image.GetViewingDirection());
            float cos_angle = ray[2] / ray.norm();
            if (cos_angle > oblique_thres) {
                valid_images_[i] = 1;
            } else {
                num_image_oblique++;
            }
        }
        std::cout << StringPrintf("Filter %d oblique images\n", num_image_oblique);

        float available_memory, est_memory;
        GetAvailableMemory(available_memory);

        ComputeTDOMDimension();

        int num_image_per_block = 0;
        std::vector<Eigen::Vector4i> blocks;
        EstimateBlocks(blocks, num_image_per_block, available_memory);

        EstimateMemoryConsume(est_memory, blocks, num_image_per_block, color_harmonization);
        std::cout << StringPrintf("Available/Est Memory: %f/%f\n", available_memory, est_memory);

        if (est_memory > available_memory) {
            std::cout << "Memory is not enough! try again after reducing gsd" << std::endl;
            ExceptionHandler(StateCode::LIMITED_CPU_MEMORY, 
                JoinPaths(GetParentDir(workspace_path_), "errors/dense"), "DenseDomGeneration").Dump();
            return ;
        }

        std::list<Triangle> triangles;
        for (auto facet : mesh_.faces_) {
            auto a = mesh_.vertices_.at(facet[0]);
            auto b = mesh_.vertices_.at(facet[1]);
            auto c = mesh_.vertices_.at(facet[2]);
            if ((a - b).norm() < 1e-6 || (b - c).norm() < 1e-6 || (a - c).norm() < 1e-6) {
                continue;
            }
            Triangle tri(Point(a[0], a[1], a[2]), Point(b[0], b[1], b[2]), 
                        Point(c[0], c[1], c[2]));
            triangles.emplace_back(tri);
        }
        std::cout << "Construct AABB Tree" << std::endl;
        // Contruct AABB tree.
        Tree tree = Tree(triangles.begin(), triangles.end());

        GetAvailableMemory(available_memory);
        std::cout << "Avaliable Memory: " << available_memory << std::endl;

        // Transform to UTM CS.
        for (auto & vtx : mesh_.vertices_) {
            vtx = trans_to_utm_ * vtx.homogeneous();
        }

        std::vector<float> image_qualities;
        std::vector<cv::Mat> bitmaps;
        image_qualities.resize(model_.images.size(), 0);
        bitmaps.resize(model_.images.size());
        std::vector<uint8_t> used_images(model_.images.size(), 0);

        const std::string dsm_path = JoinPaths(dense_reconstruction_path, "dsm.tif");
        const std::string dom_path = JoinPaths(dense_reconstruction_path, "dom.tif");

        GDALDataset* dsm_dataset = CreateGDALDataset(dsm_path, CV_32FC1, 1);
        GDALDataset* dom_dataset = CreateGDALDataset(dom_path, CV_8UC4, 4);

        GetAvailableMemory(available_memory);
        std::cout << "Avaliable Memory: " << available_memory << std::endl;

        const float downsample_factor = options_.resample_factor;
        const float inv_downsample_factor = 1.0f / downsample_factor;
        const float gsd = options_.gsd; 
        for (int block_idx = 0; block_idx < blocks.size(); ++block_idx) {
            auto block = blocks.at(block_idx);
            const int top = block[1];
            const int bottom = block[3];
            const int sub_height = bottom - top;
            top_ = top;

            const int upsample_top = top * downsample_factor;
            const int upsample_bottom = std::min(int(bottom * downsample_factor), upsample_height_);
            const int upsample_sub_height = upsample_bottom - upsample_top;
            upsample_top_ = upsample_top;

            const int left = block[0];
            const int upsample_left = left * downsample_factor;
            const int right = block[2];
            const int upsample_right = std::min(int(right * downsample_factor), upsample_width_);

            const int sub_width = right - left;
            left_ = left;

            const int upsample_sub_width = upsample_right - upsample_left;
            upsample_left_ = upsample_left;

            PrintHeading2(StringPrintf("Block %d <=> (%d %d, %d %d)", block_idx, upsample_top, upsample_left, upsample_bottom, upsample_right));

            cell_visibilities_.clear();
            cell_visibilities_.resize(sub_width * sub_height);
            // for (int k = 0; k < cell_visibilities_.size(); ++k) {
            //     cell_visibilities_.at(k).reserve(50);
            // }
            GetAvailableMemory(available_memory);
            std::cout << "Avaliable Memory: " << available_memory << std::endl;

            MatXf zBuffer(sub_width, sub_height, 1);
            zBuffer.Fill(FLT_MAX);
            CoordinateConverter coordinate_converter = {
                1.0f / gsd, x_min_, y_min_, z_min_, left, top, sub_width, sub_height
            };

            std::cout << "Render from Model" << std::endl;
            RenderFromModel(zBuffer, coordinate_converter);

            GetAvailableMemory(available_memory);
            std::cout << "Avaliable Memory: " << available_memory << std::endl;

            std::cout << "ComputeTransformDistance" << std::endl;
            {
            std::shared_ptr<MatXf> dist_maps;
            dist_maps.reset(new MatXf(sub_width, sub_height, 4));
            ComputeTransformDistance(zBuffer, *dist_maps.get());
            RefineDSM(zBuffer, *dist_maps.get());         
            dist_maps.reset();
            }

            GetAvailableMemory(available_memory);
            std::cout << "Avaliable Memory: " << available_memory << std::endl;

            Timer occ_timer;
            occ_timer.Start();

            DetectOcclusionForAllImages(zBuffer, &tree);

            std::cout << StringPrintf("Detect Occlusion cost %fmin\n", occ_timer.ElapsedMinutes());

            Timer timer;
            timer.Start();
            std::unordered_set<image_t> unique_image_ids;
            for (int y = 0; y < sub_height; ++y) {
                int pitch = y * sub_width;
                for (int x = 0; x < sub_width; ++x) {
                    auto & viss = cell_visibilities_.at(pitch + x);
                    unique_image_ids.insert(viss.begin(), viss.end());
                }
            }

            GetAvailableMemory(available_memory);
            std::cout << "Avaliable Memory: " << available_memory << std::endl;

            int released_image = 0;
            for (int image_id = 0; image_id < model_.images.size(); ++image_id) {
                if (used_images[image_id] && unique_image_ids.find(image_id) == unique_image_ids.end()) {
                    used_images[image_id] = 0;
                    bitmaps[image_id].release();
                    released_image++;
                }
            }

            std::vector<image_t> image_ids;
            image_ids.insert(image_ids.end(), unique_image_ids.begin(), unique_image_ids.end());

            int loaded_image = 0;
            for (auto image_id : image_ids) {
                if (!used_images[image_id]) {
                    loaded_image++;
                }
            }
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < image_ids.size(); ++i) {
                auto image_id = image_ids.at(i);
                if (!used_images[image_id]) {
                    used_images[image_id] = 1;
                    const mvs::Image& image = model_.images.at(image_id);
                    cv::Mat bitmap = cv::imread(image.GetPath());
                    cv::Mat gray;
                    cv::cvtColor(bitmap, gray, cv::COLOR_BGR2GRAY);
                    bitmaps[image_id] = bitmap;

                    cv::Mat fimage; 
                    gray.convertTo(fimage, CV_32FC1, 1.0 / 255, 0);
                    
                    float quality = ImageQuality(fimage);
                    image_qualities.at(image_id) = 1.0f - quality;
                    std::cout << StringPrintf("\rImage# %d quality: %f", image_id, quality);
                }
            }
            std::cout << std::endl;
            std::cout << StringPrintf("Load %d images, time cost: %fmin\n", loaded_image, timer.ElapsedMinutes());
            std::cout << StringPrintf("Release %d images, %d images are preserved\n", released_image, image_ids.size());

            GetAvailableMemory(available_memory);
            std::cout << "Avaliable Memory: " << available_memory << std::endl;

            std::shared_ptr<MatXi> holes = std::make_shared<MatXi>();
            std::unordered_map<int, int> hole_id_map;
            DetectHoles(zBuffer, *holes.get(), hole_id_map);
            FillHoles(zBuffer, *holes.get(), hole_id_map);
            hole_id_map.clear();
            holes.reset();

            std::vector<uint32_t> pixel_to_super_idx;
            std::unordered_map<uint32_t, std::vector<uint32_t> > super_idx_to_pixels;
            std::vector<std::unordered_set<uint32_t> > neighbors_per_super;
            std::vector<std::vector<uint32_t> > super_visibilities;
            std::unordered_map<image_pair_t, float> image_pairs;
            std::vector<std::vector<int> > super_ids_per_image;
            std::unordered_map<uint32_t, int> cell_labels;

            if (options_.optimizer != DOMOptimizer::DIRECTED) {
                BuildSuperPixels(zBuffer, pixel_to_super_idx, super_idx_to_pixels, neighbors_per_super, super_visibilities);

                CollectImagePairs(zBuffer, super_idx_to_pixels, super_visibilities, image_pairs, super_ids_per_image);

                if (options_.optimizer == DOMOptimizer::MAPMAP) {
                    UniGraph graph(neighbors_per_super.size());
                    InitGraphNodes(graph, pixel_to_super_idx, neighbors_per_super);

                    DataCosts data_costs(neighbors_per_super.size(), model_.images.size()); 
                    CalculateDataCosts(data_costs, image_qualities, bitmaps, zBuffer, 
                                       neighbors_per_super, super_visibilities);
                            
                    neighbors_per_super.clear();
                    super_visibilities.clear();
                    if (!color_harmonization) {
                        super_idx_to_pixels.clear();
                        super_ids_per_image.clear();
                    }

                    Optimization(data_costs, graph, cell_labels);
                }/* else if (options_.optimizer == DOMOptimizer::GCO) {
#ifdef WITH_GCO_OPTIMIZER
                    GraphOptimization(image_qualities, bitmaps, zBuffer, image_pairs, super_ids_per_image, 
                                    neighbors_per_super, super_visibilities, super_idx_to_pixels, cell_labels);
#endif
                }*/
                std::cout << "Block Mosaic" << std::endl;

                std::unordered_set<image_t> mosaic_image_ids;
                for (int r = 0; r < sub_height; ++r) {
                    for (int c = 0; c < sub_width; ++c) {
                        int super_id = pixel_to_super_idx.at(r * sub_width + c);
                        if (cell_labels.find(super_id) != cell_labels.end()) {
                            uint32_t label = cell_labels.at(super_id);
                            mosaic_image_ids.insert(label);
                        }
                    }
                }
                std::cout << "mosaic image ids: " << mosaic_image_ids.size() << std::endl;
                for (auto image_id : mosaic_image_ids) {
                    std::cout << image_id << " ";
                }
                std::cout << std::endl;
            }

            GetAvailableMemory(available_memory);
            std::cout << "Avaliable Memory: " << available_memory << std::endl;

            MatXf upsample_zBuffer(upsample_sub_width, upsample_sub_height, 1);
            upsample_zBuffer.Fill(FLT_MAX);
            CoordinateConverter coordinate_converter1 = {
                downsample_factor / gsd, x_min_, y_min_, z_min_, upsample_left_, upsample_top_, upsample_sub_width, upsample_sub_height
            };

            std::cout << "Render from Model" << std::endl;
            RenderFromModel(upsample_zBuffer, coordinate_converter1);

            GetAvailableMemory(available_memory);
            std::cout << "Avaliable Memory: " << available_memory << std::endl;

            // std::cout << "ComputeTransformDistance" << std::endl;
            // std::shared_ptr<MatXf> upsample_dist_maps;
            // upsample_dist_maps.reset(new MatXf(upsample_sub_width, upsample_sub_height, 4));
            // ComputeTransformDistance(upsample_zBuffer, *upsample_dist_maps.get());
            // RefineDSM(upsample_zBuffer, *upsample_dist_maps.get()); 

            // GetAvailableMemory(available_memory);
            // std::cout << "Avaliable Memory: " << available_memory << std::endl;

            // upsample_dist_maps.reset();

            cv::Mat dsm = cv::Mat(upsample_sub_height, upsample_sub_width, CV_32FC1, cv::Scalar(0));
            cv::Mat dom = cv::Mat(upsample_sub_height, upsample_sub_width, CV_8UC4);
            if (options_.optimizer == DOMOptimizer::DIRECTED) {
                ColorMapping(zBuffer, upsample_zBuffer, bitmaps, dom);
                for (int r = 0; r < upsample_sub_height; ++r) {
                    int downsample_r = r * inv_downsample_factor;
                    for (int c = 0; c < upsample_sub_width; ++c) {
                        int downsample_c = c * inv_downsample_factor;
                        auto & viss = cell_visibilities_.at(downsample_r * sub_width + downsample_c);
                        if (viss.empty()) {
                            continue;
                        }

                        float z = upsample_zBuffer.Get(r, c);
                        if (z >= FLT_MAX) {
                            continue;
                        }
                        Eigen::Vector3d Xd;
                        Xd[0] = (c + upsample_left_) * gsd * inv_downsample_factor + x_min_;
                        Xd[1] = (r + upsample_top_) * gsd * inv_downsample_factor + y_min_;
                        Xd[2] = z + z_min_;
                        Eigen::Vector3f X = (inv_trans_to_utm_ * Xd.homogeneous()).head<3>().cast<float>();

                        dsm.at<float>(r, c) = Xd[2] + utm_z_min_;
                        // mask.at<uchar>(r + upsample_top_, c + upsample_left_) = 255;
                    }
                }
            } else {
#pragma omp parallel for schedule(dynamic)
                for (int r = 0; r < upsample_sub_height; ++r) {
                    int downsample_r = r * inv_downsample_factor;
                    for (int c = 0; c < upsample_sub_width; ++c) {
                        int downsample_c = c * inv_downsample_factor;
                        int super_id = pixel_to_super_idx.at(downsample_r * sub_width + downsample_c);
                        if (cell_labels.find(super_id) == cell_labels.end()) {
                            continue;
                        }
                        int image_id = cell_labels.at(super_id);
                        if (image_id < 0) {
                            continue;
                        }

                        auto & viss = cell_visibilities_.at(downsample_r * sub_width + downsample_c);

                        const mvs::Image& image = model_.images.at(image_id);
                        const int image_width = image.GetWidth();
                        const int image_height = image.GetHeight();
                        Eigen::RowMatrix3x4f P(image.GetP());

                        float z = upsample_zBuffer.Get(r, c);
                        if (z >= FLT_MAX) {
                            continue;
                        }
                        Eigen::Vector3d Xd;
                        Xd[0] = (c + upsample_left_) * gsd * inv_downsample_factor + x_min_;
                        Xd[1] = (r + upsample_top_) * gsd * inv_downsample_factor + y_min_;
                        Xd[2] = z + z_min_;

                        dsm.at<float>(r, c) = Xd[2] + utm_z_min_;
                        // mask.at<uchar>(r + upsample_top_, c + upsample_left_) = 255;

                        Eigen::Vector3f X = (inv_trans_to_utm_ * Xd.homogeneous()).head<3>().cast<float>();
                        Eigen::Vector3f proj = P * X.homogeneous();
                        int u = proj.x() / proj.z();
                        int v = proj.y() / proj.z();
                        u = std::max(0, std::min(image_width - 1, u));
                        v = std::max(0, std::min(image_height - 1, v));
                        cv::Vec3b color = bitmaps.at(image_id).at<cv::Vec3b>(v, u);
                        dom.at<cv::Vec4b>(r, c) = cv::Vec4b(color[0], color[1], color[2], 255);
                    }
                }

                if (color_harmonization) {
                    std::vector<YCrCbFactor> & yrb_factors = model_.yrb_factors_;
                    if (yrb_factors.empty()) {
                        yrb_factors.resize(model_.images.size());
                        for (int image_id = 0; image_id < model_.images.size(); ++image_id) {
                            yrb_factors[image_id].s_Y = 1.0f;
                            yrb_factors[image_id].s_Cb = 1.0f;
                            yrb_factors[image_id].s_Cr = 1.0f;
                            yrb_factors[image_id].o_Y = 0.0f;
                            yrb_factors[image_id].o_Cb = 0.0f;
                            yrb_factors[image_id].o_Cb = 0.0f;
                        }
                        ColorHarmonization(zBuffer, bitmaps, image_ids, super_idx_to_pixels, image_pairs, super_ids_per_image, yrb_factors);
                    }

#pragma omp parallel for schedule(dynamic)
                    for (int r = 0; r < upsample_sub_height; ++r) {
                        int downsample_r = r * inv_downsample_factor;
                        for (int c = 0; c < upsample_sub_width; ++c) {
                            int downsample_c = c * inv_downsample_factor;
                            int super_id = pixel_to_super_idx.at(downsample_r * sub_width + downsample_c);
                            if (cell_labels.find(super_id) == cell_labels.end()) {
                                continue;
                            }
                            int image_id = cell_labels.at(super_id);
                            if (image_id < 0) {
                                continue;
                            }
                            cv::Vec4b & color = dom.at<cv::Vec4b>(r, c);

                            float V[3];
                            V[0] = color[2] * INV_COLOR_NORM;
                            V[1] = color[1] * INV_COLOR_NORM;
                            V[2] = color[0] * INV_COLOR_NORM;
                            ColorRGBToYCbCr(V);
                            V[0] = yrb_factors[image_id].s_Y * V[0] + yrb_factors[image_id].o_Y;
                            V[1] = yrb_factors[image_id].s_Cb * V[1] + yrb_factors[image_id].o_Cb;
                            V[2] = yrb_factors[image_id].s_Cr * V[2] + yrb_factors[image_id].o_Cr;

                            ColorYCbCrToRGB(V);
                            color[2] = std::min(1.0f, std::max(0.0f, V[0])) * 255;
                            color[1] = std::min(1.0f, std::max(0.0f, V[1])) * 255;
                            color[0] = std::min(1.0f, std::max(0.0f, V[2])) * 255;
                        }
                    }
                }
            }

            DumpImageMetaData(dsm_dataset, dsm);
            DumpImageMetaData(dom_dataset, dom);
        }

        GDALClose(dsm_dataset);
        GDALClose(dom_dataset);

        options_.gsd = in_gsd;
        
        timer.PrintMinutes();
    }
}

void TDOM::ReadWorkspace() {
    for (int rec_idx = 0; ;rec_idx++) {
        auto reconstruction_path = JoinPaths(workspace_path_, std::to_string(rec_idx));
        auto dense_reconstruction_path = JoinPaths(reconstruction_path, DENSE_DIR);
        auto sparse_path = JoinPaths(dense_reconstruction_path, SPARSE_DIR);
        auto fused_path = JoinPaths(dense_reconstruction_path, FUSION_NAME);
        auto model_path = JoinPaths(dense_reconstruction_path, MODEL_NAME);
        if (!ExistsDir(reconstruction_path)) {
            break;
        }
        if (!ExistsDir(dense_reconstruction_path)) {
            std::cout << StringPrintf("dense path is not exist!\n");
            break;
        }
        if (!ExistsFile(fused_path)) {
            std::cout << StringPrintf("fused.ply is not exist!\n");
            break;
        }
        if (!ExistsFile(model_path)) {
            std::cout << StringPrintf("model.obj is not exist!\n");
            break;
        }
        num_reconstruction_++;
    }
    std::cout << "Reading workspace (" << num_reconstruction_ << " reconstructions)..." << std::endl;
}

void TDOM::SetUp(int rec_idx) {
    auto reconstruction_path = JoinPaths(workspace_path_, std::to_string(rec_idx));
    auto dense_reconstruction_path = JoinPaths(reconstruction_path, DENSE_DIR);
    // auto sparse_path = JoinPaths(dense_reconstruction_path, SPARSE_DIR);
    auto undistort_image_path = JoinPaths(dense_reconstruction_path, IMAGES_DIR);
    auto fused_path = JoinPaths(dense_reconstruction_path, FUSION_NAME);
    auto fused_vis_path = JoinPaths(dense_reconstruction_path, FUSION_NAME) + ".vis";
    auto model_path = JoinPaths(dense_reconstruction_path, MODEL_NAME);

    mesh_.Clear();
    ReadTriangleMeshObj(model_path, mesh_, true);
    mesh_.vertex_colors_.clear();
    mesh_.vertex_normals_.clear();
    mesh_.vertex_labels_.clear();
    mesh_.vertex_status_.clear();
    mesh_.vertex_visibilities_.clear();
    mesh_.face_normals_.clear();

    // double fDecimate = 0.3;
    // mesh_.Clean(fDecimate, 0, false, 0, 0, false);

    mvs::Workspace::Options workspace_options;
    workspace_options.image_as_rgb = true;
    workspace_options.workspace_path = dense_reconstruction_path;
    workspace_options.workspace_format = "perspective";
    workspace_options.image_path = JoinPaths(dense_reconstruction_path, IMAGES_DIR);

    std::unique_ptr<mvs::Workspace> workspace;
    workspace.reset(new mvs::Workspace(workspace_options));
    model_ = workspace->GetModel();
}

void TDOM::ComputeTransformationToUTM() {
    std::vector<Eigen::Vector3d> src_points;
    std::vector<Eigen::Vector3d> tgt_points;
    src_points.reserve(mesh_.vertices_.size());
    tgt_points.reserve(mesh_.vertices_.size());

    int zone, sourth_or_north = 0;
    double latitude, longitude, altitude;
    for (auto vtx : mesh_.vertices_) {
        src_points.push_back(vtx);
        Eigen::Vector3d Xw = trans_ * vtx.homogeneous();
        GPSReader::LocationToGps(Xw, &latitude, &longitude, &altitude);
        Eigen::Vector3d utm = GPSReader::gpsToUTM(latitude, longitude);
        tgt_points.emplace_back(utm.x(), utm.y(), altitude);
        zone = (int)utm.z();
        sourth_or_north = (latitude >= 0 ? 1 : -1);
    }

    sourth_or_north_ = sourth_or_north;
    zone_no_ = zone;
    // utm_zone_ = MapToUTMZone(zone, sourth_or_north);
    std::cout << "UTM Zone: " << zone_no_ << std::endl;

    Eigen::Matrix<double, 3, Eigen::Dynamic> src_mat(3, src_points.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst_mat(3, tgt_points.size());
    for (size_t i = 0; i < src_points.size(); ++i) {
        src_mat.col(i) = src_points[i];
        dst_mat.col(i) = tgt_points[i];
    }

    Eigen::RowMatrix3x4d model;
    model = Eigen::umeyama(src_mat, dst_mat, false).topLeftCorner(3, 4);

    x_min_ = y_min_ = z_min_ = std::numeric_limits<double>::max();
    x_max_ = y_max_ = z_max_ = std::numeric_limits<double>::lowest();
    for (auto vtx : mesh_.vertices_) {
        vtx = model * vtx.homogeneous();
        x_min_ = std::min(x_min_, vtx[0]);
        y_min_ = std::min(y_min_, vtx[1]);
        z_min_ = std::min(z_min_, vtx[2]);
        x_max_ = std::max(x_max_, vtx[0]);
        y_max_ = std::max(y_max_, vtx[1]);
        z_max_ = std::max(z_max_, vtx[2]);
    }

    utm_x_min_ = x_min_;
    utm_y_min_ = y_min_;
    utm_z_min_ = z_min_;

    model(0, 3) -= x_min_;
    model(1, 3) -= y_min_;
    model(2, 3) -= z_min_;
    trans_to_utm_ = model;

    x_max_ -= x_min_;
    y_max_ -= y_min_;
    z_max_ -= z_min_;
    x_min_ = y_min_ = z_min_ = 0;

    Eigen::RowMatrix4d htrans = Eigen::RowMatrix4d::Identity();
    htrans.block<3, 4>(0, 0) = trans_to_utm_;
    inv_trans_to_utm_ = htrans.inverse();

    std::cout << "Transformation To UTM Coordinate System: " << std::endl << trans_to_utm_ << std::endl;
    std::cout << "Inverse: " << std::endl << inv_trans_to_utm_ << std::endl;
}

void TDOM::EstimateGSD() {
    // Estimate GSD.
    std::cout << "Estimate GSD..." << std::endl;

    m_track_length_ = 0;
    std::vector<std::vector<size_t> > images_points;
    images_points.resize(model_.images.size());
    for (size_t k = 0; k < model_.points.size(); ++k) {
        auto point = model_.points.at(k);
        for (auto image_id : point.track) {
            images_points.at(image_id).push_back(k);
        }
        m_track_length_ += point.track.size();
    }
    m_track_length_ /= model_.points.size();
    std::cout << "Mean Track Length: " << m_track_length_ << std::endl;

    float m_focal = 0.0f;
    std::vector<float> heights;
    heights.reserve(images_points.size());
    for (int i = 0; i < images_points.size(); ++i) {
        auto & image_points = images_points.at(i);
        if (image_points.size() == 0) {
            continue;
        }
        mvs::Image image = model_.images.at(i);
        Eigen::RowMatrix3x4f P(image.GetP());
        m_focal += image.GetK()[0];

        std::vector<float> image_heights;
        image_heights.reserve(image_points.size());
        for (size_t point_idx = 0; point_idx < image_points.size(); ++point_idx) {
            Eigen::Vector3f X(&model_.points.at(point_idx).x);
            Eigen::Vector3f Xc = (P * X.homogeneous());
            image_heights.push_back(Xc[2]);
        }
        int nth = image_heights.size() / 2;
        std::nth_element(image_heights.begin(), image_heights.begin() + nth, image_heights.end());
        heights.push_back(image_heights[nth]);
    }

    int nth = heights.size() / 2;
    std::nth_element(heights.begin(), heights.begin() + nth, heights.end());
    float m_height = heights.at(nth);
    m_focal /= heights.size();
    est_gsd_ = m_height / m_focal;
    float est_gsd;
    if (options_.gsd <= 0) {
        est_gsd = 2.0f * est_gsd_; // 2 times greater than the estimate.
    } else {
        est_gsd = options_.gsd; // 2 times greater than the estimate.
    }
    // est_gsd = (int)std::floor(est_gsd * 100.0f) / 100.0f;
    options_.gsd = options_.resample_factor * est_gsd;
    std::cout << "gsd: " << m_height << " " << m_focal << " " << est_gsd << std::endl;
    std::cout << "resample gsd: " << options_.gsd << std::endl;
}

void TDOM::ComputeTDOMDimension() {
    const float inv_gsd = 1.0f / options_.gsd;
    width_ = (x_max_ - x_min_) * inv_gsd + 1;
    height_ = (y_max_ - y_min_) * inv_gsd + 1;
    upsample_width_ = (x_max_ - x_min_) * inv_gsd * options_.resample_factor + 1;
    upsample_height_ = (y_max_ - y_min_) * inv_gsd * options_.resample_factor + 1;

    std::cout << StringPrintf("Range: [%f %f %f, %f %f %f]\n", x_min_, y_min_, z_min_, x_max_, y_max_, z_max_);
    std::cout << StringPrintf("DSM resolution: %d %d\n", width_, height_);
    std::cout << StringPrintf("Upsample DSM resolution: %d %d\n", upsample_width_, upsample_height_);
}

void TDOM::EstimateBlocks(std::vector<Eigen::Vector4i> & blocks, int & num_image_per_block, const double available_memory) {
    std::cout << "Split Blocks" << std::endl;
    const float inv_gsd = 1.0f / options_.gsd;
    size_t max_width = 0, max_height = 0;

    std::vector<Eigen::Vector2i> proj_coords;
    proj_coords.reserve(model_.images.size());
    for (int image_id = 0; image_id < model_.images.size(); ++image_id) {
        if (!valid_images_.at(image_id)) {
            continue;
        }
        mvs::Image & image = model_.images.at(image_id);
        max_width = std::max(image.GetWidth(), max_width);
        max_height = std::max(image.GetHeight(), max_height);

        Eigen::Vector3d C(image.GetC()[0], image.GetC()[1], image.GetC()[2]);
        C = trans_to_utm_ * C.homogeneous();
        Eigen::Vector2i xy;
        xy.x() = (C[0] - x_min_) * inv_gsd;
        xy.y() = (C[1] - y_min_) * inv_gsd;
        proj_coords.push_back(xy);
    }
    if (proj_coords.empty()) {
        return;
    }
    double est_memory = 1e-9 * max_width * max_height * model_.images.size() * 3;
    if (available_memory < 0.5) {
        ExceptionHandler(LIMITED_CPU_MEMORY, 
            JoinPaths(GetParentDir(workspace_path_), "errors/dense"), "DenseDomGeneration").Dump();
        exit(StateCode::LIMITED_CPU_MEMORY);
    }

    int num_blocks = 2 * est_memory / available_memory + 1;
    num_image_per_block = (model_.images.size() + num_blocks - 1) / num_blocks;

    double model_memory = 1e-9 * (mesh_.vertices_.size() * 8 * 3 * 2);
    double image_memory = 1e-9 * max_width * max_height * 3 * num_image_per_block;
    double buf_memory = 1e-9 * upsample_width_ * upsample_height_ * 4 * (6 + m_track_length_ * 2) +
                        1e-9 * upsample_width_ * upsample_height_ * 3 * 4;
    double total_memory = model_memory + image_memory + buf_memory;
    int num_blocks_spatial = 1;
    while(total_memory > available_memory) {
        buf_memory /= 2;
        num_blocks_spatial *= 2;
        total_memory = model_memory + image_memory + buf_memory;
    }
    num_blocks = std::max(num_blocks, num_blocks_spatial);
    std::cout << "Estimate Blocks: " << num_blocks << std::endl;

    num_image_per_block = (model_.images.size() + num_blocks - 1) / num_blocks;
    std::cout << "num images per block: " << num_image_per_block << std::endl;

    Eigen::Vector4i range(0, 0, width_, height_);
    int axis = (width_ > height_ ? 0 : 1);
    SplitSpace(proj_coords, num_image_per_block, axis, range, blocks, 0);

    // cv::Mat img(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));
    // cv::RNG rng(time(0));
    // for (auto box : blocks) {
    //     // cv::Rect rect(box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1);
    //     cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    //     cv::rectangle(img, cv::Point2f(box[0], box[1]), cv::Point2f(box[2], box[3]), color, -1);
    // }
    // for (auto coord : proj_coords) {
    //     cv::circle(img, cv::Point(coord.x(), coord.y()), 8, cv::Scalar(0, 0, 255));
    // }
    // cv::imwrite("./split.jpg", img);
}

void TDOM::EstimateMemoryConsume(float & est_memory, const std::vector<Eigen::Vector4i> &blocks, 
                                 const int num_image_per_block, const bool color_harmonization) {
    size_t max_image_width = 0, max_image_height = 0;
    for (auto & image : model_.images) {
        max_image_width = std::max(image.GetWidth(), max_image_width);
        max_image_height = std::max(image.GetHeight(), max_image_height);
    }

    const float downsample_factor = options_.resample_factor;

    int max_sub_width(0), max_sub_height(0);
    int max_upsample_sub_width(0), max_upsample_sub_height(0);
    for (int block_idx = 0; block_idx < blocks.size(); ++block_idx) {
        auto block = blocks.at(block_idx);
        const int top = block[1];
        const int bottom = block[3];
        const int sub_height = bottom - top;
        const int upsample_top = top * downsample_factor;
        const int upsample_bottom = std::min(int(bottom * downsample_factor), upsample_height_);
        const int upsample_sub_height = upsample_bottom - upsample_top;

        const int left = block[0];
        const int right = block[2];
        const int sub_width = right - left;
        const int upsample_left = left * downsample_factor;
        const int upsample_right = std::min(int(right * downsample_factor), upsample_width_);
        const int upsample_sub_width = upsample_right - upsample_left;

        max_sub_width = std::max(max_sub_width, sub_width);
        max_sub_height = std::max(max_sub_height, sub_height);
        max_upsample_sub_width = std::max(max_upsample_sub_width, upsample_sub_width);
        max_upsample_sub_height = std::max(max_upsample_sub_height, upsample_sub_height);
    }

    double model_memory = 1e-9 * (mesh_.vertices_.size() * 8 * 3 * 2);
    double image_memory = 1e-9 * max_image_width * max_image_height * 3 * num_image_per_block;
    double buf_memory = 1e-9 * max_sub_width * max_sub_height * 4 * (6 + m_track_length_ * 2) +
                        1e-9 * max_upsample_sub_width * max_upsample_sub_height * 3 * 4;
    if (color_harmonization) {
        buf_memory += 1e-9 * max_upsample_sub_width * max_upsample_sub_height * 4;
    }

    est_memory = (model_memory + image_memory + buf_memory) * 1.15;
}

void TDOM::RenderFromModel(MatXf & zBuffer, CoordinateConverter& coordinate_converter) {
    const int width = zBuffer.GetWidth();
    const int height = zBuffer.GetHeight();
#pragma omp parallel for schedule(dynamic)
    for (int face_id = 0; face_id < mesh_.faces_.size(); ++face_id) {
        auto facet = mesh_.faces_.at(face_id);
        Eigen::Vector3d & vtx0 = mesh_.vertices_.at(facet[2]);
        Eigen::Vector3d & vtx1 = mesh_.vertices_.at(facet[1]);
        Eigen::Vector3d & vtx2 = mesh_.vertices_.at(facet[0]);

        Eigen::Vector3f i0 = coordinate_converter(vtx0.x(), vtx0.y(), vtx0.z());
        Eigen::Vector3f i1 = coordinate_converter(vtx1.x(), vtx1.y(), vtx1.z());
        Eigen::Vector3f i2 = coordinate_converter(vtx2.x(), vtx2.y(), vtx2.z());

        int u_min = std::min(i0[0], std::min(i1[0], i2[0]));
        int u_max = std::max(i0[0], std::max(i1[0], i2[0]));
        int v_min = std::min(i0[1], std::min(i1[1], i2[1]));
        int v_max = std::max(i0[1], std::max(i1[1], i2[1]));
        if (u_max < 0 || u_min >= coordinate_converter.sub_width || 
            v_max < 0 || v_min >= coordinate_converter.sub_height) {
            continue;
        }
        u_min = std::max(0, u_min);
        v_min = std::max(0, v_min);
        u_max = std::min(width - 1, u_max);
        v_max = std::min(height - 1, v_max);

        float x0, y0, z0, x1, y1, z1, x2, y2, z2;
        x0 = i0.x(); y0 = i0.y(); z0 = i0.z();
        x1 = i1.x(); y1 = i1.y(); z1 = i1.z();
        x2 = i2.x(); y2 = i2.y(); z2 = i2.z();

        float norm1 = -(x0 - x1) * (y2 - y1) + (y0 - y1) * (x2 - x1);
        float norm2 = -(x1 - x2) * (y0 - y2) + (y1 - y2) * (x0 - x2);

        Eigen::Vector2d v1(x1 - x0, y1 - y0);
        Eigen::Vector2d v2(x2 - x1, y2 - y1);
        Eigen::Vector2d v3(x0 - x2, y0 - y2);

        for (int v = v_min; v <= v_max; ++v) {
            for (int u = u_min; u <= u_max; ++u) {
                double m1 = v1.x() * (v - y0) - (u - x0) * v1.y();
                double m2 = v2.x() * (v - y1) - (u - x1) * v2.y();
                double m3 = v3.x() * (v - y2) - (u - x2) * v3.y();
                float a = (-(u - x1) * (y2 - y1) + (v - y1) * (x2 - x1)) / norm1;
                float b = (-(u - x2) * (y0 - y2) + (v - y2) * (x0 - x2)) / norm2;
                if (a < 0 || a > 1 || b < 0 || b > 1 || a + b > 1) {
                    continue;
                }

                float z = a * z0 + b * z1 + (1 - a - b) * z2;
#pragma omp critical
                {
                    float z_old = zBuffer.Get(v, u);
                    if (z_old >= FLT_MAX || z_old < z) {
                        zBuffer.Set(v, u, z);
                    }
                }
            }
        }
    }
}

void TDOM::ComputeTransformDistance(const MatXf & zBuffer, MatXf & dist_maps) {
    // Initialization.
    const int width = zBuffer.GetWidth();
    const int height = zBuffer.GetHeight();
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (zBuffer.Get(i, j) < FLT_MAX) {
                dist_maps.Set(i, j, 0, 0);
                dist_maps.Set(i, j, 1, 0);
                dist_maps.Set(i, j, 2, 0);
                dist_maps.Set(i, j, 3, 0);
            } else {
                dist_maps.Set(i, j, 0, MAX_DIST);
                dist_maps.Set(i, j, 1, MAX_DIST);
                dist_maps.Set(i, j, 2, MAX_DIST);
                dist_maps.Set(i, j, 3, MAX_DIST);
            }
        }
    }

    // Compute col dist.
    float n_dist, nd_tmp;
    for (int i = 0; i < height; ++i) {
        for (int j = 1; j < width; ++j) {
            n_dist = dist_maps.Get(i, j, 0);
            if (n_dist != 0 && (nd_tmp = 1 + dist_maps.Get(i, j - 1, 0)) < n_dist) {
                dist_maps.Set(i, j, 0, nd_tmp);
            }
        }
        for (int j = width - 2; j >=0; --j) {
            n_dist = dist_maps.Get(i, j, 2);
            if (n_dist != 0 && (nd_tmp = 1 + dist_maps.Get(i, j + 1, 2)) < n_dist) {
                dist_maps.Set(i, j, 2, nd_tmp);
            }
        }
    }

    // Compute row dist.
    for (int j = 0; j < width; ++j) {
        for (int i = 1; i < height; ++i) {
            n_dist = dist_maps.Get(i, j, 1);
            if (n_dist != 0 && (nd_tmp = 1 + dist_maps.Get(i - 1, j, 1)) < n_dist) {
                dist_maps.Set(i, j, 1, nd_tmp);
            }
        }
        for (int i = height - 2; i >= 0; --i) {
            n_dist = dist_maps.Get(i, j, 3);
            if (n_dist != 0 && (nd_tmp = 1 + dist_maps.Get(i + 1, j, 3)) < n_dist) {
                dist_maps.Set(i, j, 3, nd_tmp);
            }
        }
    }
}

void TDOM::RefineDSM(MatXf & zBuffer, const MatXf & dist_maps) {

    std::cout << "RefineDSM" << std::endl;

    const int width = zBuffer.GetWidth();
    const int height = zBuffer.GetHeight();

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (zBuffer.Get(i, j) < FLT_MAX) {
                continue;
            }
            float dist0 = dist_maps.Get(i, j, 0);
            float dist1 = dist_maps.Get(i, j, 1);
            float dist2 = dist_maps.Get(i, j, 2);
            float dist3 = dist_maps.Get(i, j, 3);

            int counter = 0;
            float weight = 0.0f;
            float est_depth = 0.0f;
            if (dist0 != MAX_DIST && j - dist0 >= 0) {
                float depth0 = zBuffer.Get(i, j - dist0);
                est_depth += depth0 / dist0;
                weight += 1.0f / dist0;
                counter++;
            }
            if (dist2 != MAX_DIST && j + dist2 < width) {
                float depth2 = zBuffer.Get(i, j + dist2);
                est_depth += depth2 / dist2;
                weight += 1.0f / dist2;
                counter++;
            }
            if (dist1 != MAX_DIST && i - dist1 >= 0) {
                float depth1 = zBuffer.Get(i - dist1, j);
                est_depth += depth1 / dist1;
                weight += 1.0f / dist1;
                counter++;
            }
            if (dist3 != MAX_DIST && i + dist3 < height) {
                float depth3 = zBuffer.Get(i + dist3, j);
                est_depth += depth3 / dist3;
                weight += 1.0f / dist3;
                counter++;
            }
            if (counter > 2) {
                est_depth /= weight;
                zBuffer.Set(i, j, est_depth);
            }
        }
    }
}

void TDOM::ConvertToPseudoColor(const cv::Mat & fimage, cv::Mat & out_image, const cv::Mat & mask) {

    std::cout << "ConvertToPseudoColor" << std::endl;
    out_image = cv::Mat(fimage.rows, fimage.cols, CV_8UC4);

    const float robust_max = z_max_; // - 0.02 * (z_max_ - z_min_);
    const float robust_min = z_min_; // + 0.02 * (z_max_ - z_min_);
    const float robust_range = robust_max - robust_min;

#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < fimage.rows; ++y) {
        const uchar *ptr = mask.ptr<uchar>(y);
        for (int x = 0; x < fimage.cols; ++x, ++ptr) {
            const float d1 = fimage.at<float>(y, x);
            if (*ptr == 0) {
                out_image.at<cv::Vec4b>(y, x) = cv::Vec4b(0, 0, 0, 0);
                continue;
            }

            const float robust_val = std::max(robust_min, std::min(robust_max, float(d1 - z_min_ - utm_z_min_)));
            // const float gray = 1.0f - d1 / (z_max_ - z_min_);
            const float gray = 1.0f - robust_val / robust_range;
            const BitmapColor<float> color(
                255 * JetColormap::Red(gray),
                255 * JetColormap::Green(gray),
                255 * JetColormap::Blue(gray));
            out_image.at<cv::Vec4b>(y, x) = cv::Vec4b(color.r, color.g, color.b, 255);
        }
    }
}

void TDOM::WriteGeoText(const cv::Mat & dsm, const cv::Mat & mask, std::ofstream &file) {
    const int width = dsm.cols;
    const int height = dsm.rows;
    const float gsd = options_.gsd;

    int dump_lx = -1, dump_ly = -1;
    int dump_rx = -1, dump_ry = -1;
    int validate_x = -1, validate_y = -1;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (mask.at<uint8_t>(y, x) == 0) {
                continue;
            }
            if (dump_lx != -1 && dump_rx != -1 && validate_x == -1) {
                validate_x = x; validate_y = y;
            }

            if (dump_lx == -1) {
                dump_lx = x; dump_ly = y;
            }
            if (dump_rx == -1) {
                dump_rx = width - x - 1; dump_ry = height - y - 1;
            }
            if (dump_lx != -1 && dump_rx != -1 && validate_x != -1) {
                break;
            }
        }
    }

    file << "# Image dimension: WIDTH HEIGHT" << std::endl;
    file << width << " " << height << std::endl;
    file << "# Ground Sample Distance" << std::endl;
    file << gsd << std::endl;
    if (dump_lx != -1 && dump_rx != -1 && validate_x != -1) {
        Eigen::Vector3d X, Xw, Xd;
        double latitude, longitude, altitude;

        const float inv_factor = 1.0f / options_.resample_factor;
        // first point.
        Xd[0] = dump_lx * gsd * inv_factor + x_min_;
        Xd[1] = (height - 1 - dump_ly) * gsd * inv_factor + y_min_;
        Xd[2] = dsm.at<float>(dump_ly, dump_lx) + z_min_;
        X = (inv_trans_to_utm_ * Xd.homogeneous()).head<3>();

        Xw = trans_ * X.homogeneous();
        GPSReader::LocationToGps(Xw, &latitude, &longitude, &altitude);
        file << "# First point" << std::endl;
        file << dump_lx << " " << dump_ly << std::endl;
        file << MAX_PRECISION << latitude << " " << longitude << " " << altitude << std::endl;

        // second point.
        Xd[0] = dump_rx * gsd * inv_factor + x_min_;
        Xd[1] = (height - 1 - dump_ry) * gsd * inv_factor + y_min_;
        Xd[2] = dsm.at<float>(dump_ry, dump_rx) + z_min_;
        X = (inv_trans_to_utm_ * Xd.homogeneous()).head<3>();

        Xw = trans_ * X.homogeneous();
        GPSReader::LocationToGps(Xw, &latitude, &longitude, &altitude);
        file << "# Second point" << std::endl;
        file << dump_rx << " " << dump_ry << std::endl;
        file << MAX_PRECISION << latitude << " " << longitude << " " << altitude << std::endl;

        // validate point.
        Xd[0] = validate_x * gsd * inv_factor + x_min_;
        Xd[1] = (height - 1 - validate_y) * gsd * inv_factor + y_min_;
        Xd[2] = dsm.at<float>(height - 1 - validate_y, validate_x) + z_min_;
        X = (inv_trans_to_utm_ * Xd.homogeneous()).head<3>();

        Xw = trans_ * X.homogeneous();
        GPSReader::LocationToGps(Xw, &latitude, &longitude, &altitude);
        file << "# Validate point" << std::endl;
        file << validate_x << " " << validate_y << std::endl;
        file << MAX_PRECISION << latitude << " " << longitude << " " << altitude << std::endl;
    }
}

GDALDataset* TDOM::CreateGDALDataset(const std::string &filename, const int image_type, const int channel) {
    GDALAllRegister();

    const char *pszFormat = "GTiff";
    GDALDriver *driver;
    driver = GetGDALDriverManager()->GetDriverByName(pszFormat);
    if (!driver) {
        std::cout << "Get Driver by name failed!" << std::endl;
        return NULL;
    }

    char** papszMetadata = driver->GetMetadata();;
    if (!CSLFetchBoolean(papszMetadata, GDAL_DCAP_CREATE, FALSE)) {
        fprintf(stderr, "Driver %s don't supports Create() method.\n", pszFormat);
        return NULL;
    }

    char **papszOptions = NULL;
    papszOptions = CSLSetNameValue(papszOptions, "NUM_THREADS", std::to_string(GetEffectiveNumThreads(-1)).c_str());
    papszOptions = CSLSetNameValue(papszOptions, "BIGTIFF", "YES");
    papszOptions = CSLSetNameValue(papszOptions, "COMPRESS", "DEFLATE");
    // papszOptions = CSLSetNameValue(papszOptions, "COMPRESS", "LZW");
    // papszOptions = CSLSetNameValue(papszOptions, "INTERLEAVE", "BAND");
    papszOptions = CSLSetNameValue(papszOptions, "PROFILE", "GeoTIFF");
    if (image_type == CV_8UC3) {
        papszOptions = CSLSetNameValue(papszOptions, "PHOTOMETRIC", "RGB");
        // papszOptions = CSLSetNameValue(papszOptions, "NBITS", "8");
    } else if (image_type == CV_8UC4) {
        papszOptions = CSLSetNameValue(papszOptions, "PHOTOMETRIC", "RGB");
        papszOptions = CSLSetNameValue(papszOptions, "ALPHA", "YES");
        // papszOptions = CSLSetNameValue(papszOptions, "NBITS", "8");
    } else if (image_type == CV_32FC1) {
        papszOptions = CSLSetNameValue(papszOptions, "PHOTOMETRIC", "MINISBLACK");
        papszOptions = CSLSetNameValue(papszOptions, "NBITS", "16");
    }

    GDALDataset *pdataset;
    if (image_type == CV_8UC3 || image_type == CV_8UC4) {
        pdataset = driver->Create(filename.c_str(), upsample_width_, upsample_height_, channel, GDT_Byte, papszOptions);
    } else if (image_type == CV_32FC1) {
        pdataset = driver->Create(filename.c_str(), upsample_width_, upsample_height_, 1, GDT_Float32, papszOptions);
    }
    if (!pdataset) {
        std::cout << "Create Dataset Failed!" << std::endl << std::flush;
    } else {
        std::cout << "Create Dataset Success" << std::endl << std::flush;
    }

    const double gsd = options_.gsd;
    const double scale = options_.gsd / options_.resample_factor;

    double origin_x, origin_y;
    origin_x = x_min_ + utm_x_min_;
    origin_y = (upsample_height_ - 1) * scale + y_min_ + utm_y_min_;

    double geo_transform[6] = {origin_x, scale, 0, origin_y, 0, -scale};
    pdataset->SetGeoTransform(geo_transform);

    OGRSpatialReference oSRS;
    oSRS.SetWellKnownGeogCS("WGS84");
    oSRS.SetUTM(zone_no_, (sourth_or_north_ > 0));
    char *pszWKT = NULL;
    oSRS.exportToWkt(&pszWKT);
    pdataset->SetProjection(pszWKT);
    return pdataset;
}

void TDOM::DumpImageMetaData(GDALDataset *pdataset, const cv::Mat & image) {
    const int top_y = upsample_height_ - 1 - upsample_top_;
    if (image.type() == CV_32FC1 || image.type() == CV_64FC1) {
        for (int r = 0; r < image.rows; ++r) {
            const float *ptr = image.ptr<float>(r);
            pdataset->GetRasterBand(1)->RasterIO(GF_Write, upsample_left_, top_y - r, image.cols, 1, 
                                                (void *)ptr, image.cols, 1, GDT_Float32, 4, 4 * image.cols);
        }
    } else if (image.type() == CV_8UC3) {
        std::vector<uint8_t> buf1(image.cols, 0);
        std::vector<uint8_t> buf2(image.cols, 0);
        std::vector<uint8_t> buf3(image.cols, 0);
        for (int r = 0; r < image.rows; ++r) {
            for (int c = 0; c < image.cols; ++c) {
                const cv::Vec3b & color = image.at<cv::Vec3b>(r, c);
                buf1[c] = color[2]; 
                buf2[c] = color[1]; 
                buf3[c] = color[0]; 
            }
            pdataset->GetRasterBand(1)->RasterIO(GF_Write, upsample_left_, top_y - r, image.cols, 1, 
                                                 buf1.data(), image.cols, 1, GDT_Byte, 1, image.cols);
            pdataset->GetRasterBand(2)->RasterIO(GF_Write, upsample_left_, top_y - r, image.cols, 1, 
                                                 buf2.data(), image.cols, 1, GDT_Byte, 1, image.cols);
            pdataset->GetRasterBand(3)->RasterIO(GF_Write, upsample_left_, top_y - r, image.cols, 1, 
                                                 buf3.data(), image.cols, 1, GDT_Byte, 1, image.cols);
        }
    } else if (image.type() == CV_8UC4) {
        std::vector<uint8_t> buf1(image.cols, 0);
        std::vector<uint8_t> buf2(image.cols, 0);
        std::vector<uint8_t> buf3(image.cols, 0);
        std::vector<uint8_t> buf4(image.cols, 0);
        for (int r = 0; r < image.rows; ++r) {
            for (int c = 0; c < image.cols; ++c) {
                const cv::Vec4b & color = image.at<cv::Vec4b>(r, c);
                buf1[c] = color[2]; 
                buf2[c] = color[1]; 
                buf3[c] = color[0]; 
                buf4[c] = color[3];
            }
            pdataset->GetRasterBand(1)->RasterIO(GF_Write, upsample_left_, top_y - r, image.cols, 1, 
                                                 buf1.data(), image.cols, 1, GDT_Byte, 1, image.cols);
            pdataset->GetRasterBand(2)->RasterIO(GF_Write, upsample_left_, top_y - r, image.cols, 1, 
                                                 buf2.data(), image.cols, 1, GDT_Byte, 1, image.cols);
            pdataset->GetRasterBand(3)->RasterIO(GF_Write, upsample_left_, top_y - r, image.cols, 1, 
                                                 buf3.data(), image.cols, 1, GDT_Byte, 1, image.cols);
            pdataset->GetRasterBand(4)->RasterIO(GF_Write, upsample_left_, top_y - r, image.cols, 1, 
                                                 buf4.data(), image.cols, 1, GDT_Byte, 1, image.cols);
        }

        // std::cout << "Uint8_t * 4" << std::endl;
        // pdataset->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, image.cols, image.rows, buf1.data(), image.cols, image.rows, GDT_Byte, 1, image.cols);
        // pdataset->GetRasterBand(2)->RasterIO(GF_Write, 0, 0, image.cols, image.rows, buf2.data(), image.cols, image.rows, GDT_Byte, 1, image.cols);
        // pdataset->GetRasterBand(3)->RasterIO(GF_Write, 0, 0, image.cols, image.rows, buf3.data(), image.cols, image.rows, GDT_Byte, 1, image.cols);
        // pdataset->GetRasterBand(4)->RasterIO(GF_Write, 0, 0, image.cols, image.rows, buf4.data(), image.cols, image.rows, GDT_Byte, 1, image.cols);

        // int panBandMap[4] = {1, 2, 3, 4};
        // CPLErr error = pdataset->RasterIO(GF_Write, 0, 0, image.cols, image.rows, buf.data(), image.cols, image.rows, GDT_Byte, 4, panBandMap, 1, image.cols, image.cols * image.rows);
        // if (error != CE_None) {
        //     std::cout << error << std::endl;
        // }
    }
    // GDALClose(pdataset);
}

void TDOM::DetectHoles(const MatXf & zBuffer, MatXi & holes, std::unordered_map<int, int> & hole_id_map) {
    const int width = zBuffer.GetWidth();
    const int height = zBuffer.GetHeight();
    int hole_id = 1;
    holes = MatXi(width, height, 1);
    holes.Fill(0);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (!cell_visibilities_.at(i * width + j).empty() || holes.Get(i, j)) {
                continue;
            }
            bool is_border = false;
            int num_connected_comp = 0;
            std::queue<std::pair<int, int> > Q;
            Q.push(std::make_pair(j, i));
            holes.Set(i, j, hole_id);
            while(!Q.empty()) {
                std::pair<int, int> coord = Q.front();
                Q.pop();
                num_connected_comp++;
                for (int k = 0; k < 4; ++k) {
                    int r = coord.second + dirs[k][0];
                    int c = coord.first + dirs[k][1];
                    if (r < 0 || r >= height || c < 0 || c >= width) {
                        continue;
                    }
                    if (!cell_visibilities_.at(r * width + c).empty() || holes.Get(r, c)) {
                        continue;
                    }
                    holes.Set(r, c, hole_id);
                    Q.push(std::make_pair(c, r));
                    if (r == 0 || c == 0 || r == height - 1 || c == width - 1) {
                        is_border = true;
                    }
                }
            }
            if (is_border) {
                hole_id_map[hole_id] = -1;
            } else {
                hole_id_map[hole_id] = num_connected_comp;
            }
            hole_id++;
        }
    }
}

void TDOM::DetectOcclusionForAllImages(const MatXf & zBuffer, const tdom::Tree * tree) {
    const int sub_width = zBuffer.GetWidth();
    const int sub_height = zBuffer.GetHeight();
    const float gsd = options_.gsd;
    const double estep = 0.05;
    const float cos_thres = std::cos(DEG2RAD(60));

#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < sub_height; ++y) {
        for (int x = 0; x < sub_width; ++x) {
            float z = zBuffer.Get(y, x);
            if (z >= FLT_MAX) {
                continue;
            }

            Eigen::Vector3d Xd;
            Xd[0] = (x + left_) * gsd + x_min_;
            Xd[1] = (y + top_) * gsd + y_min_;
            Xd[2] = z + z_min_;
            Eigen::Vector3f X = (inv_trans_to_utm_ * Xd.homogeneous()).head<3>().cast<float>();

            auto & cell_visibility = cell_visibilities_.at(y * sub_width + x);

            std::vector<std::pair<int, float> > weighted_visibilities;
            weighted_visibilities.reserve(model_.images.size());

            for (int image_id = 0; image_id < model_.images.size(); ++image_id) {
                if (!valid_images_.at(image_id)) {
                    continue;
                }
                const mvs::Image& image = model_.images.at(image_id);
                Eigen::RowMatrix3x4f P(image.GetP());
                Eigen::Vector3f C(image.GetC());
                Eigen::Vector3f ray(image.GetR() + 6);

                const int image_width = image.GetWidth();
                const int image_height = image.GetHeight();

                Eigen::Vector3f proj = P * X.homogeneous();
                int u = proj.x() / proj.z();
                int v = proj.y() / proj.z();
                if (u < 0 || u >= image_width || v < 0 || v >= image_height) {
                    continue;
                }
                
                Eigen::Vector3f point_ray = (X - C).normalized();
                float cos_angle = point_ray.dot(ray);
                if (cos_angle < cos_thres) {
                    continue;
                }

                // Construct segment query.
                Eigen::Vector3f query_point = X - point_ray * estep;
                Point a(query_point[0], query_point[1], query_point[2]);
                Point b(C[0], C[1], C[2]);
                Segment segment_query(a, b);

                // Test intersection with segment query.
                if (tree->do_intersect(segment_query)) {
                    continue;
                }

                weighted_visibilities.emplace_back(image_id, cos_angle);
            }
            int nth = std::min(MAX_CAPACITY, (int)weighted_visibilities.size() - 1);
            std::nth_element(weighted_visibilities.begin(), weighted_visibilities.begin() + nth, weighted_visibilities.end(),
                [&](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                    return a.second > b.second;
                });
            for (int k = 0; k < nth; ++k) {
                cell_visibility.push_back(weighted_visibilities[k].first);
            }
            std::sort(cell_visibility.begin(), cell_visibility.end());
        }
        if (y % 100 == 0) {
            std::cout << StringPrintf("\rrows: %d/%d", y, sub_height) << std::flush;
        }
    }
    std::cout << std::endl;
}

void TDOM::FillHoles(const MatXf & zBuffer, MatXi & holes, std::unordered_map<int, int> & hole_id_map) {
    Timer timer;
    timer.Start();
    
    const int sub_width = zBuffer.GetWidth();
    const int sub_height = zBuffer.GetHeight();
    const float gsd = options_.gsd;
    const float cos_thres = std::cos(DEG2RAD(60));

    MatXf vis_Buffer(sub_width, sub_height, 1);
    for (int r = 0; r < sub_height; ++r) {
        for (int c = 0; c < sub_width; ++c) {
            auto & viss = cell_visibilities_.at(r * sub_width + c);
            if (viss.empty()) {
                vis_Buffer.Set(r, c, FLT_MAX);
            } else {
                vis_Buffer.Set(r, c, 0.0f);
            }
        }
    }

    std::shared_ptr<MatXf> dist_maps;
    dist_maps.reset(new MatXf(sub_width, sub_height, 4));
    ComputeTransformDistance(vis_Buffer, *dist_maps.get());
    RefineDSM(vis_Buffer, *dist_maps.get());

    int max_image_width = 0;
    int max_image_height = 0;
    for (int i = 0; i < model_.images.size(); ++i) {
        max_image_width = std::max(max_image_width, (int)model_.images.at(i).GetWidth());
        max_image_height = std::max(max_image_height, (int)model_.images.at(i).GetHeight());
    }
    const int max_len = std::min(max_image_width, max_image_height) * 0.25 * est_gsd_ * options_.resample_factor / options_.gsd;
    // std::cout << "max_len: " << max_len << std::endl;

    // #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < sub_height; ++i) {
        for (int j = 0; j < sub_width; ++j) {
            int hole_id = holes.Get(i, j);
            if (hole_id == 0) {
                continue;
            }
            int num_connected_comp = hole_id_map.at(hole_id);
            if (num_connected_comp == -1 || num_connected_comp >= 250000) {
                continue;
            }
            float z = zBuffer.Get(i, j);
            if (z >= FLT_MAX) {
                continue;
            }

            Eigen::Vector3d Xd;
            Xd[0] = (j + left_) * gsd + x_min_;
            Xd[1] = (i + top_) * gsd + y_min_;
            Xd[2] = z + z_min_;
            Eigen::Vector3f X = (inv_trans_to_utm_ * Xd.homogeneous()).head<3>().cast<float>();

            float dist0 = dist_maps->Get(i, j, 0);
            float dist1 = dist_maps->Get(i, j, 1);
            float dist2 = dist_maps->Get(i, j, 2);
            float dist3 = dist_maps->Get(i, j, 3);

            int counter = 0;
            std::unordered_set<uint32_t> nviss;
            if (dist0 < max_len && j - dist0 >= 0) {
                counter++;
                auto & viss = cell_visibilities_.at(i * sub_width + j - dist0);
                nviss.insert(viss.begin(), viss.end());
            }
            if (dist2 < max_len && j + dist2 < sub_width) {
                counter++;
                auto & viss = cell_visibilities_.at(i * sub_width + j + dist2);
                nviss.insert(viss.begin(), viss.end());
            }
            if (dist1 < max_len && i - dist1 >= 0) {
                counter++;
                auto & viss = cell_visibilities_.at((i - dist1) * sub_width + j);
                nviss.insert(viss.begin(), viss.end());
            }
            if (dist3 < max_len && i + dist3 < sub_height) {
                counter++;
                auto & viss = cell_visibilities_.at((i + dist3) * sub_width + j);
                nviss.insert(viss.begin(), viss.end());
            }
            if (counter > 2) {
                std::vector<std::pair<int, float> > weighted_visibilities;
                for (auto image_id : nviss) {
                    const mvs::Image& image = model_.images.at(image_id);
                    Eigen::RowMatrix3x4f P(image.GetP());
                    Eigen::Vector3f C(image.GetC());
                    Eigen::Vector3f ray(image.GetR() + 6);

                    const int image_width = image.GetWidth();
                    const int image_height = image.GetHeight();

                    Eigen::Vector3f proj = P * X.homogeneous();
                    int u = proj.x() / proj.z();
                    int v = proj.y() / proj.z();
                    if (u < 0 || u >= image_width || v < 0 || v >= image_height) {
                        continue;
                    }
                    Eigen::Vector3f point_ray = (X - C).normalized();
                    float cos_angle = point_ray.dot(ray);
                    if (cos_angle < cos_thres) {
                        continue;
                    }
                    weighted_visibilities.emplace_back(image_id, cos_angle);
                }
                int nth = std::min(3, (int)weighted_visibilities.size());
                if (nth < 0) {
                    continue;
                }
                std::nth_element(weighted_visibilities.begin(), weighted_visibilities.begin() + nth, weighted_visibilities.end(),
                    [&](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                        return a.second > b.second;
                    });
                for (int k = 0; k < nth; ++k) {
                    cell_visibilities_.at(i * sub_width + j).push_back(weighted_visibilities[k].first);
                }
            }
        }
    }
    std::cout << StringPrintf("Fill Holes cost %fmin\n", timer.ElapsedMinutes());
}

void TDOM::ColorMapping(const MatXf & zBuffer, const MatXf & upsample_zBuffer, 
                        const std::vector<cv::Mat> &bitmaps, cv::Mat & dom) {
    Timer timer;
    timer.Start();

    const int sub_width = zBuffer.GetWidth();
    const int sub_height = zBuffer.GetHeight();
    const int upsample_sub_width = upsample_zBuffer.GetWidth();
    const int upsample_sub_height = upsample_zBuffer.GetHeight();
    const float downsample_factor = options_.resample_factor;
    const float inv_downsample_factor = 1.0f / downsample_factor;
    const float gsd = options_.gsd;

#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < upsample_sub_height; ++y) {
        int downsample_r = y * inv_downsample_factor;
        if (downsample_r >= sub_height) {
            continue;
        }
        for (int x = 0; x < upsample_sub_width; ++x) {
            int downsample_c = x * inv_downsample_factor;
            if (downsample_c >= sub_width) {
                continue;
            }
            auto & viss = cell_visibilities_.at(downsample_r * sub_width + downsample_c);
            if (viss.empty()) {
                dom.at<cv::Vec4b>(y, x) = cv::Vec4b(0, 0, 0, 0);
                continue;
            }

            float z = upsample_zBuffer.Get(y, x);
            if (z >= FLT_MAX) {
                dom.at<cv::Vec4b>(y, x) = cv::Vec4b(0, 0, 0, 0);
                continue;
            }
            Eigen::Vector3d Xd;
            Xd[0] = (x + upsample_left_) * gsd * inv_downsample_factor + x_min_;
            Xd[1] = (y + upsample_top_) * gsd * inv_downsample_factor + y_min_;
            Xd[2] = z + z_min_;
            Eigen::Vector3f X = (inv_trans_to_utm_ * Xd.homogeneous()).head<3>().cast<float>();
            
            std::vector<uint8_t> fused_r, fused_g, fused_b;
            fused_r.reserve(viss.size());
            fused_g.reserve(viss.size());
            fused_b.reserve(viss.size());
            for (auto image_id : viss) {
                const mvs::Image& image = model_.images.at(image_id);

                Eigen::RowMatrix3x4f P(image.GetP());
                Eigen::Vector3f C(image.GetC());
                Eigen::Vector3f ray(image.GetR() + 6);

                const int image_width = image.GetWidth();
                const int image_height = image.GetHeight();

                Eigen::Vector3f proj = P * X.homogeneous();
                int u = proj.x() / proj.z();
                int v = proj.y() / proj.z();

                if (u < 0 || u >= image_width || v < 0 || v >= image_height) {
                    continue;
                }

                cv::Vec3b color = bitmaps[image_id].at<cv::Vec3b>(v, u);
                fused_r.push_back(color[2]);
                fused_g.push_back(color[1]);
                fused_b.push_back(color[0]);
            }
            if (fused_r.empty()) {
                dom.at<cv::Vec4b>(y, x) = cv::Vec4b(0, 0, 0, 0);
                continue;
            }
            size_t nth = fused_r.size() / 2;
            std::nth_element(fused_r.begin(), fused_r.begin() + nth, fused_r.end());
            std::nth_element(fused_g.begin(), fused_g.begin() + nth, fused_g.end());
            std::nth_element(fused_b.begin(), fused_b.begin() + nth, fused_b.end());
            dom.at<cv::Vec4b>(y, x) = cv::Vec4b(fused_b[nth], fused_g[nth], fused_r[nth], 255);
        }
    }
    std::cout << StringPrintf("Color Mapping cost %fmin\n", timer.ElapsedMinutes());
}

void TDOM::CollectImagePairs(const MatXf & zBuffer, 
                            const std::unordered_map<uint32_t, std::vector<uint32_t> > & super_idx_to_pixels,
                            const std::vector<std::vector<uint32_t> > & super_visibilities,
                            std::unordered_map<image_pair_t, float> & image_pairs,
                            std::vector<std::vector<int> > & super_ids_per_image) {
    Timer timer;
    timer.Start();

    std::cout << "Collect Image Pairs" << std::endl;
    const int num_images = model_.images.size();
    const int sub_width = zBuffer.GetWidth();
    const int sub_height = zBuffer.GetHeight();
    const uint32_t image_size = sub_width * sub_height;
    const int num_super_pixel = super_visibilities.size();

    super_ids_per_image.resize(num_images);
    for (int i = 0; i < num_images; ++i) {
        super_ids_per_image.at(i).reserve(num_super_pixel);
    }

    for (int super_id = 1; super_id < num_super_pixel; ++super_id) {
        auto & image_ids = super_visibilities.at(super_id);
        for (int i = 0; i < image_ids.size(); ++i) {
            image_t image_id1 = image_ids.at(i);
            super_ids_per_image.at(image_id1).push_back(super_id);
            for (int j = i + 1; j < image_ids.size(); ++j) {
                image_t image_id2 = image_ids.at(j);
                image_pair_t pair_id = utility::ImagePairToPairId(image_id1, image_id2);
                image_pairs[pair_id] += 1.0f;
            }
        }
    }
    std::vector<image_pair_t> image_pair_ids;
    image_pair_ids.reserve(image_pairs.size());
    for (auto image_pair : image_pairs) {
        image_pair_ids.push_back(image_pair.first);
    }

#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < image_pair_ids.size(); ++k) {
        const auto image_pair_id = image_pair_ids.at(k);
        image_t image_id1;
        image_t image_id2;
        utility::PairIdToImagePair(image_pair_id, &image_id1, &image_id2);
        auto super_ids1 = super_ids_per_image.at(image_id1);
        auto super_ids2 = super_ids_per_image.at(image_id2);

        std::vector<int> inter_super_ids;
        std::set_intersection(super_ids1.begin(), super_ids1.end(), super_ids2.begin(), super_ids2.end(), 
                              std::back_inserter(inter_super_ids));

        std::unordered_set<int> unique_pixels;
        for (auto super_id : inter_super_ids) {
            auto & pixels_per_super = super_idx_to_pixels.at(super_id);
            for (auto pixel : pixels_per_super) {
                unique_pixels.insert(pixel);
            }
        }
        float area = unique_pixels.size() * 1.0f / image_size;
        image_pairs.at(image_pair_id) = area;
    }
    std::cout << StringPrintf("%d image pairs\n", image_pairs.size());
    timer.PrintMinutes();
}

void TDOM::ColorHarmonization(const MatXf & zBuffer, const std::vector<cv::Mat> & bitmaps,
                              const std::vector<image_t> & unique_image_ids,
                              const std::unordered_map<uint32_t, std::vector<uint32_t> > & super_idx_to_pixels,
                              const std::unordered_map<image_pair_t, float> & image_pairs,
                              const std::vector<std::vector<int> > & super_ids_per_image,
                              std::vector<YCrCbFactor> & yrb_factors) {
    // const int num_images = model_.images.size();
    const int num_images = unique_image_ids.size();
    const float downsample_factor = options_.resample_factor;
    const float gsd = options_.gsd;

    const int sub_width = zBuffer.GetWidth();
    const int sub_height = zBuffer.GetHeight();
    const uint32_t image_size = sub_width * sub_height;

    std::cout << "Graph Edge Compression" << std::endl;

    std::vector<std::pair<image_pair_t, float> > sorted_image_pairs;
    sorted_image_pairs.reserve(image_pairs.size());

    MinimumSpanningTree<image_t, float> mst_extractor;

    std::vector<image_pair_t> image_pair_ids;
    image_pair_ids.reserve(image_pairs.size());
    for (auto image_pair : image_pairs) {
        image_t image_id1;
        image_t image_id2;
        utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
        image_pair_ids.push_back(image_pair.first);
        
        mst_extractor.AddEdge(image_id1, image_id2, -image_pair.second);
        sorted_image_pairs.emplace_back(image_pair.first, image_pair.second);
    }
    std::unordered_set<std::pair<image_t, image_t> > minimum_spanning_tree;
    mst_extractor.Extract(&minimum_spanning_tree);

    const int num_redundance_edges = 0.4 * (image_pairs.size() - num_images + 1);
    std::nth_element(sorted_image_pairs.begin(), sorted_image_pairs.begin() + num_redundance_edges + num_images - 1, 
                     sorted_image_pairs.end(), 
        [&](const std::pair<image_pair_t, float>& a, const std::pair<image_pair_t, float>& b) {
            return a.second > b.second;
        });

    image_pair_ids.clear();
    image_pair_ids.reserve(image_pairs.size());
    std::unordered_set<image_pair_t> tree_image_pair_ids; 
    for (auto image_pair : minimum_spanning_tree) {
        auto pair_id = utility::ImagePairToPairId(image_pair.first, image_pair.second);
        image_pair_ids.push_back(pair_id);
        tree_image_pair_ids.insert(pair_id);
    }
    int num_edge = 0;
    while(num_edge < num_redundance_edges) {
        auto pair_id = sorted_image_pairs.at(num_edge).first;
        if (tree_image_pair_ids.find(pair_id) == tree_image_pair_ids.end()) {
            image_pair_ids.push_back(pair_id);
        }
        num_edge++;
    }

    std::sort(image_pair_ids.begin(), image_pair_ids.end());

    std::cout << StringPrintf("%d edges are preserved, %d edges are pruned!\n",
        image_pair_ids.size(), image_pairs.size() - image_pair_ids.size());

    std::cout << "Color Harmonization" << std::endl;

    ceres::Problem problem_Y, problem_Cb, problem_Cr;
    ceres::LossFunction *loss_function_Y = new ceres::SoftLOneLoss(1.0);
    ceres::LossFunction *loss_function_Cb = new ceres::SoftLOneLoss(1.0);
    ceres::LossFunction *loss_function_Cr = new ceres::SoftLOneLoss(1.0);

    int num_image_pair = 0;

#pragma omp parallel for schedule(dynamic)
     for (int k = 0; k < image_pair_ids.size(); ++k) {
        auto image_pair_id = image_pair_ids.at(k);
        image_t image_id1;
        image_t image_id2;
        utility::PairIdToImagePair(image_pair_id, &image_id1, &image_id2);

        const mvs::Image& image1 = model_.images.at(image_id1);
        const mvs::Image& image2 = model_.images.at(image_id2);
        const int image_width1 = image1.GetWidth();
        const int image_height1 = image1.GetHeight();
        const int image_width2 = image2.GetWidth();
        const int image_height2 = image2.GetHeight();
        Eigen::RowMatrix3x4f P1(image1.GetP());
        Eigen::RowMatrix3x4f P2(image2.GetP());

        const cv::Mat & bitmap1 = bitmaps.at(image_id1);
        const cv::Mat & bitmap2 = bitmaps.at(image_id2);

        auto super_ids1 = super_ids_per_image.at(image_id1);
        auto super_ids2 = super_ids_per_image.at(image_id2);

        Histogram hist1_Y(0.0f, 1.0f, 50);
        Histogram hist1_Cb(0.0f, 1.0f, 50);
        Histogram hist1_Cr(0.0f, 1.0f, 50);
        Histogram hist2_Y(0.0f, 1.0f, 50);
        Histogram hist2_Cb(0.0f, 1.0f, 50);
        Histogram hist2_Cr(0.0f, 1.0f, 50);

        std::vector<int> inter_super_ids;
        std::set_intersection(super_ids1.begin(), super_ids1.end(), super_ids2.begin(), super_ids2.end(), 
                              std::back_inserter(inter_super_ids));

        std::unordered_set<int> unique_pixels;
        for (auto super_id : inter_super_ids) {
            auto & pixels_per_super = super_idx_to_pixels.at(super_id);
            for (auto pixel : pixels_per_super) {
                unique_pixels.insert(pixel);
            }
        }
        std::vector<int> pixels;
        pixels.reserve(unique_pixels.size());
        pixels.insert(pixels.end(), unique_pixels.begin(), unique_pixels.end());

        // float area = pixels.size() * 1.0f / (sub_width * sub_height);
        // if (area < 0.03) continue;

        // std::cout << "area: " << pixels.size() << "/" << sub_width * sub_height 
        //           << " " << pixels.size() * 1.0f / (sub_width * sub_height) << std::endl;
        for (auto pixel_id : pixels) {
            int r = pixel_id / sub_width;
            int c = pixel_id % sub_width;

            float z = zBuffer.Get(r, c);
            if (z >= FLT_MAX) {
                continue;
            }
            Eigen::Vector3d Xd;
            Xd[0] = (c + left_) * gsd + x_min_;
            Xd[1] = (r + top_) * gsd + y_min_;
            Xd[2] = z + z_min_;
            Eigen::Vector3f X = (inv_trans_to_utm_ * Xd.homogeneous()).head<3>().cast<float>();

            Eigen::Vector3f proj1 = P1 * X.homogeneous();
            int u1 = proj1.x() / proj1.z();
            int v1 = proj1.y() / proj1.z();
            if (u1 < 0 || u1 >= image_width1 || v1 < 0 || v1 >= image_height1) {
                continue;
            }
            cv::Vec3b color1 = bitmap1.at<cv::Vec3b>(v1, u1);
            float V1[3];
            V1[0] = color1[2] * INV_COLOR_NORM;
            V1[1] = color1[1] * INV_COLOR_NORM;
            V1[2] = color1[0] * INV_COLOR_NORM;
            ColorRGBToYCbCr(V1);

            Eigen::Vector3f proj2 = P2 * X.homogeneous();
            int u2 = proj2.x() / proj2.z();
            int v2 = proj2.y() / proj2.z();
            if (u2 < 0 || u2 >= image_width2 || v2 < 0 || v2 >= image_height2) {
                continue;
            }
            cv::Vec3b color2 = bitmap2.at<cv::Vec3b>(v2, u2);
            float V2[3];
            V2[0] = color2[2] * INV_COLOR_NORM;
            V2[1] = color2[1] * INV_COLOR_NORM;
            V2[2] = color2[0] * INV_COLOR_NORM;
            ColorRGBToYCbCr(V2);

            hist1_Y.add_value(V1[0]);
            hist1_Cb.add_value(V1[1]);
            hist1_Cr.add_value(V1[2]);
            hist2_Y.add_value(V2[0]);
            hist2_Cb.add_value(V2[1]);
            hist2_Cr.add_value(V2[2]);
        }
        #pragma omp critical
        {
        float bi = 0.01f;
        while (bi < 1.0f) {

            ceres::CostFunction *cost_function_Y = nullptr, *cost_function_Cb = nullptr, *cost_function_Cr = nullptr;

            cost_function_Y = ColorCorrectionCostFunction::Create(
                    hist1_Y.get_approx_percentile(bi),
                    hist2_Y.get_approx_percentile(bi));
            problem_Y.AddResidualBlock(cost_function_Y, loss_function_Y, &yrb_factors[image_id1].s_Y,
                                    &yrb_factors[image_id2].s_Y, &yrb_factors[image_id1].o_Y, &yrb_factors[image_id2].o_Y);

            cost_function_Cb = ColorCorrectionCostFunction::Create(
                    hist1_Cb.get_approx_percentile(bi),
                    hist2_Cb.get_approx_percentile(bi));
            problem_Cb.AddResidualBlock(cost_function_Cb, loss_function_Cb, &yrb_factors[image_id1].s_Cb,
                                        &yrb_factors[image_id2].s_Cb, &yrb_factors[image_id1].o_Cb, &yrb_factors[image_id2].o_Cb);

            cost_function_Cr = ColorCorrectionCostFunction::Create(
                    hist1_Cr.get_approx_percentile(bi),
                    hist2_Cr.get_approx_percentile(bi));
            problem_Cr.AddResidualBlock(cost_function_Cr, loss_function_Cr, &yrb_factors[image_id1].s_Cr,
                                        &yrb_factors[image_id2].s_Cr, &yrb_factors[image_id1].o_Cr, &yrb_factors[image_id2].o_Cr);

            bi += 0.02f;
        }
        num_image_pair++;
        std::cout << StringPrintf("\rProcess image pair: [%6d, %6d] %d/%d", image_id1, image_id2, num_image_pair, image_pair_ids.size()) << std::flush;
        // std::cout << StringPrintf("Process image pair: [%6d, %6d] %d/%d\n", image_id1, image_id2, num_image_pair, image_pair_ids.size()) << std::flush;
        }
    }

#pragma omp parallel for schedule(dynamic)
    for(std::size_t i = 0; i < num_images; ++i){
        image_t image_id = unique_image_ids[i];
        if(problem_Y.HasParameterBlock(&yrb_factors[image_id].s_Y)) {
            problem_Y.SetParameterLowerBound(&yrb_factors[image_id].s_Y, 0, 1 - 0.4);
            problem_Y.SetParameterUpperBound(&yrb_factors[image_id].s_Y, 0, 1 + 0.4);
            problem_Y.SetParameterLowerBound(&yrb_factors[image_id].o_Y, 0, -30.0 / 255);
            problem_Y.SetParameterUpperBound(&yrb_factors[image_id].o_Y, 0, 30.0 / 255);
        }
        if(problem_Cb.HasParameterBlock(&yrb_factors[image_id].s_Cb)) {
            problem_Cb.SetParameterLowerBound(&yrb_factors[image_id].s_Cb, 0, 1 - 0.2);
            problem_Cb.SetParameterUpperBound(&yrb_factors[image_id].s_Cb, 0, 1 + 0.2);
            problem_Cb.SetParameterLowerBound(&yrb_factors[image_id].o_Cb, 0, -5.0 / 255);
            problem_Cb.SetParameterUpperBound(&yrb_factors[image_id].o_Cb, 0, 5.0 / 255);
        }
        if(problem_Cr.HasParameterBlock(&yrb_factors[image_id].s_Cr)) {
            problem_Cr.SetParameterLowerBound(&yrb_factors[image_id].s_Cr, 0, 1 - 0.2);
            problem_Cr.SetParameterUpperBound(&yrb_factors[image_id].s_Cr, 0, 1 + 0.2);
            problem_Cr.SetParameterLowerBound(&yrb_factors[image_id].o_Cr, 0, -5.0 / 255);
            problem_Cr.SetParameterUpperBound(&yrb_factors[image_id].o_Cr, 0, 5.0 / 255);
        }
    }

    std::cout << "Start Graph Optimization" << std::endl;

    ceres::Solver::Options solver_options;
    solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    solver_options.num_threads = GetEffectiveNumThreads(-1);

    ceres::Solver::Summary summary_Y;
    ceres::Solve(solver_options, &problem_Y, &summary_Y);
    printf("Solve channel Y\n");
    PrintSolverSummary(summary_Y);

    ceres::Solver::Summary summary_Cb;
    ceres::Solve(solver_options, &problem_Cb, &summary_Cb);
    printf("Solve channel Cb\n");
    PrintSolverSummary(summary_Cb);

    ceres::Solver::Summary summary_Cr;
    ceres::Solve(solver_options, &problem_Cr, &summary_Cr);
    printf("Solve channel Cr\n");
    PrintSolverSummary(summary_Cr);

    for (auto i : unique_image_ids) {
        std::cout << StringPrintf("Local Luminance Adjust(%d): scale (%f %f %f), offset (%f %f %f)", i, 
            yrb_factors[i].s_Y, yrb_factors[i].s_Cb, yrb_factors[i].s_Cr, 
            yrb_factors[i].o_Y, yrb_factors[i].o_Cb, yrb_factors[i].o_Cr) << std::endl;
    }
}

void TDOM::BuildSuperPixels(const MatXf & zBuffer, std::vector<uint32_t>& pixel_to_super_idx,
                            std::unordered_map<uint32_t, std::vector<uint32_t> > & super_idx_to_pixels,
                            std::vector<std::unordered_set<uint32_t> >& neighbors_per_super,
                            std::vector<std::vector<uint32_t> >& super_visibilities) {
    Timer timer;
    timer.Start();

    const int sub_width = zBuffer.GetWidth();
    const int sub_height = zBuffer.GetHeight();

    int max_super_id = 0;
    super_idx_to_pixels[max_super_id].clear();
    pixel_to_super_idx.resize(sub_width * sub_height);
    std::fill(pixel_to_super_idx.begin(), pixel_to_super_idx.end(), 0);

    for (int r = 0; r < sub_height; ++r) {
        for (int c = 0; c < sub_width; ++c) {
            uint32_t cid = r * sub_width + c;
            if (cell_visibilities_.at(cid).empty() || pixel_to_super_idx.at(cid) != 0) {
                continue;
            }
            pixel_to_super_idx.at(cid) = ++max_super_id;
            super_idx_to_pixels[max_super_id].push_back(cid);
            
            std::queue<uint32_t> Q;
            Q.push(cid);
            while(!Q.empty()) {
                uint32_t cid = Q.front();
                Q.pop();

                auto c_viss = cell_visibilities_.at(cid);

                std::ostringstream oss;
                std::copy(c_viss.begin(), c_viss.end(), std::ostream_iterator<int>(oss, ""));
                std::string cvis_str = oss.str(); 

                int y = cid / sub_width;
                int x = cid % sub_width;
                for (int i = 0; i < 8; ++i) {
                    int ny = y + dirs[i][0];
                    int nx = x + dirs[i][1];
                    if (ny < 0 || ny >= sub_height || nx < 0 || nx >= sub_width) {
                        continue;
                    }
                    uint32_t nid = ny * sub_width + nx;
                    auto n_viss = cell_visibilities_.at(nid);
                    if (n_viss.empty() || pixel_to_super_idx.at(nid) != 0) {
                        continue;
                    }

                    std::ostringstream oss;
                    std::copy(n_viss.begin(), n_viss.end(), std::ostream_iterator<int>(oss, ""));
                    std::string nvis_str = oss.str(); 
                    if (nvis_str.compare(cvis_str) == 0) {
                        pixel_to_super_idx.at(nid) = pixel_to_super_idx.at(cid);
                        super_idx_to_pixels[max_super_id].push_back(nid);
                        Q.push(nid);
                    }
                }
            }

        }
    }
    std::cout << "Number of super pixel: " << max_super_id << std::endl;

    neighbors_per_super.resize(max_super_id + 1);
    super_visibilities.resize(max_super_id + 1);

    // Find super neighborhood.
    std::cout << "Find neighborhood of super pixels." << std::endl;
    
    std::vector<bool> visited(sub_height * sub_width);
    std::fill(visited.begin(), visited.end(), false);

    std::queue<uint32_t> Q;
    Q.push(0);
    visited[0] = true;
    while(!Q.empty()) {
        uint32_t cid = Q.front();
        Q.pop();
        
        uint32_t super_id = pixel_to_super_idx.at(cid);
        if (super_visibilities.at(super_id).empty() && !cell_visibilities_.at(cid).empty()) {
            super_visibilities.at(super_id) = cell_visibilities_.at(cid);
        }

        int y = cid / sub_width;
        int x = cid % sub_width;
        for (int i = 0; i < 8; ++i) {
            int ny = y + dirs[i][0];
            int nx = x + dirs[i][1];
            uint32_t nid = ny * sub_width + nx;
            if (ny < 0 || ny >= sub_height || nx < 0 || nx >= sub_width || visited[nid]) {
                continue;
            }
            if (super_id != pixel_to_super_idx[nid]) {
                neighbors_per_super.at(super_id).insert(pixel_to_super_idx[nid]);
                neighbors_per_super.at(pixel_to_super_idx[nid]).insert(super_id);
            }
            Q.push(nid);
            visited[nid] = true;
        }
    }

#if 0
    std::cout << "Merge isolated super pixels" << std::endl;

    const int min_num_pixel_per_super = 25;
    int nth = 0;
    std::vector<std::pair<int, int> > sorted_supers;
    sorted_supers.reserve(neighbors_per_super.size());
    for (int super_id = 0; super_id < neighbors_per_super.size(); ++super_id) {
        int num_pixel_per_super = super_idx_to_pixels.at(super_id).size();
        if (num_pixel_per_super < min_num_pixel_per_super) {
            nth++;
        }
        sorted_supers.emplace_back(super_id, num_pixel_per_super);
    }

    std::cout << "nth: " << nth << std::endl;

    std::nth_element(sorted_supers.begin(), sorted_supers.begin() + nth, sorted_supers.end(),
        [&](const std::pair<int, int> &a, const std::pair<int, int> &b) {
            return a.second < b.second;
        });

    int merged_super = 0;
    visited.clear();
    visited.resize(neighbors_per_super.size(), false);
    std::vector<int> parent_ids(neighbors_per_super.size());
    for (int super_id = 0; super_id < parent_ids.size(); ++super_id) {
        parent_ids[super_id] = super_id;
    }
    for (int i = 0; i < nth; ++i) {
        auto super_id = sorted_supers.at(i).first;
        visited[super_id] = true;
        auto & neighbor_ids = neighbors_per_super.at(super_id);
        auto & pixels = super_idx_to_pixels.at(super_id);
        if (pixels.size() > min_num_pixel_per_super) {
            break;
        }

        auto & viss = super_visibilities.at(super_id);

        int merge_neighbor_id = -1;
        float similar_ratio = 0.0f;
        for (auto neighbor_id : neighbor_ids) {
            if (visited[neighbor_id]) {
                continue;
            }

            auto & nviss = super_visibilities.at(neighbor_id);
            std::vector<uint32_t> res;
            std::set_intersection(viss.begin(), viss.end(), nviss.begin(), nviss.end(), std::back_inserter(res));
            float ratio = std::min(1.0f * res.size() / viss.size(), 1.0f * res.size() / nviss.size());
            if (ratio > similar_ratio) {
                similar_ratio = ratio;
                merge_neighbor_id = neighbor_id;
            }
        }
        if (similar_ratio > 0.8 && merge_neighbor_id != -1) {
            merged_super++;
            parent_ids[super_id] = merge_neighbor_id;
        }
    }

    for (int super_id = 0; super_id < parent_ids.size(); ++super_id) {
        if (super_id == parent_ids[super_id]) {
            continue;
        }
        int parent_id = parent_ids.at(super_id);
        while(parent_id != parent_ids[parent_id]) {
            parent_id = parent_ids[parent_id];
        }

        auto & pixels = super_idx_to_pixels.at(super_id);
        auto & merge_pixles = super_idx_to_pixels.at(parent_id);
        merge_pixles.insert(merge_pixles.end(), pixels.begin(), pixels.end());
        for (auto pixel_id : pixels) {
            pixel_to_super_idx.at(pixel_id) = parent_id;
        }
        pixels.clear();

        auto & neighbor_ids = neighbors_per_super.at(super_id);
        for (auto neighbor_id : neighbor_ids) {
            neighbors_per_super.at(neighbor_id).erase(super_id);
        }
        neighbor_ids.erase(parent_id);
        neighbors_per_super.at(parent_id).insert(neighbor_ids.begin(), neighbor_ids.end());
        neighbor_ids.clear();
    }

    std::cout << StringPrintf("Merge %d/%d supers\n", merged_super, neighbors_per_super.size());
#endif
    std::cout << StringPrintf("Construct Super Pixels cost %fmin\n", timer.ElapsedMinutes());
}

void TDOM::InitGraphNodes(UniGraph & graph, const std::vector<uint32_t>& pixel_to_super_idx,
                          const std::vector<std::unordered_set<uint32_t> >& neighbors_per_super) {

    std::cout << "Initializing Graph Nodes" << std::endl;

    int num_super_pixel = neighbors_per_super.size();
    for (int super_id = 1; super_id < num_super_pixel; ++super_id) {
        for (auto neighbor_id : neighbors_per_super[super_id]) {
            if (!graph.has_edge(super_id, neighbor_id)) {
                graph.add_edge(super_id, neighbor_id);
            }
        }
    }
}

void TDOM::CalculateDataCosts(DataCosts & data_costs, const std::vector<float> & image_qualities,
                              const std::vector<cv::Mat> & bitmaps, const MatXf & zBuffer,
                              const std::vector<std::unordered_set<uint32_t> >& neighbors_per_super,
                              const std::vector<std::vector<uint32_t> >& super_visibilities/*,
                              std::unordered_map<uint32_t, std::vector<uint32_t> > & super_idx_to_pixels*/) {
    
    std::cout << "CalculateDataCosts" << std::endl;

    const int sub_width = zBuffer.GetWidth();
    const int sub_height = zBuffer.GetHeight();
    const float gsd = options_.gsd;

    if (options_.optimizer == DOMOptimizer::MAPMAP) {
        printf("mapmap solver\n");
#pragma omp parallel for schedule(dynamic)
        for (int super_id = 1; super_id < super_visibilities.size(); ++super_id) {
            auto & image_ids = super_visibilities.at(super_id);
            std::vector<std::pair<uint32_t, float> > image_scores;
            image_scores.reserve(image_ids.size());
            for (auto & image_id : image_ids) {
                image_scores.emplace_back(image_id, image_qualities.at(image_id));
            }

            // int max_num_label = std::min((int)image_ids.size(), 5);
            int max_num_label = image_ids.size();
            std::nth_element(image_scores.begin(), image_scores.begin() + max_num_label, image_scores.end(),
                [](const std::pair<uint32_t, float> &s1, const std::pair<uint32_t, float> &s2) {
                    return s1.second < s2.second;
                });

            std::vector<std::pair<uint32_t, float> > image_scores_sorted;
            image_scores_sorted.resize(max_num_label);
            for (int k = 0; k < max_num_label; ++k) {
                image_scores_sorted[k] = image_scores.at(k);
            }

            std::sort(image_scores_sorted.begin(), image_scores_sorted.end(), 
                [&](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
                    return a.first < b.first;
                });

#pragma omp critical
            for (int k = 0; k < max_num_label; ++k) {
                uint32_t image_id = image_scores_sorted.at(k).first;
                float score = image_scores_sorted.at(k).second;
                data_costs.set_value(super_id, image_id, score);
            }
        }
    }/* else if (options_.optimizer == DOMOptimizer::GCO) {
        printf("GCO solver\n");
#pragma omp parallel for schedule(dynamic)
        for (int super_id = 1; super_id < super_visibilities.size(); ++super_id) {
            auto pixel_idx_per_super = super_idx_to_pixels.at(super_id);
            if (pixel_idx_per_super.empty()) {
                continue;
            }

            auto & image_ids = super_visibilities.at(super_id);
            std::vector<std::pair<uint32_t, float> > image_scores;
            image_scores.reserve(image_ids.size());
            for (auto & image_id : image_ids) {
                image_scores.emplace_back(image_id, image_qualities.at(image_id));
            }

            // int max_num_label = std::min((int)image_ids.size(), 5);
            int max_num_label = image_ids.size();
            std::nth_element(image_scores.begin(), image_scores.begin() + max_num_label, image_scores.end(),
                [](const std::pair<uint32_t, float> &s1, const std::pair<uint32_t, float> &s2) {
                    return s1.second < s2.second;
                });

            std::vector<std::pair<uint32_t, float> > image_scores_sorted;
            image_scores_sorted.resize(max_num_label);
            for (int k = 0; k < max_num_label; ++k) {
                image_scores_sorted[k] = image_scores.at(k);
            }

            std::sort(image_scores_sorted.begin(), image_scores_sorted.end(), 
                [&](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
                    return a.first < b.first;
                });

#pragma omp critical
            for (int k = 0; k < max_num_label; ++k) {
                uint32_t image_id = image_scores_sorted.at(k).first;
                float score = image_scores_sorted.at(k).second;
                data_costs.set_value(super_id, image_id, score);
            }
        }
    }*/
}

void TDOM::Optimization(const DataCosts & data_costs, UniGraph & graph, 
                        std::unordered_map<uint32_t, int> & cell_labels) {
    using uint_t = unsigned int;
    using cost_t = float;
    constexpr uint_t simd_w = mapmap::sys_max_simd_width<cost_t>();
    using unary_t = mapmap::UnaryTable<cost_t, simd_w>;
    using pairwise_t = mapmap::PairwisePotts<cost_t, simd_w>;
    
    // Construct graph
    mapmap::Graph<cost_t> mgraph(graph.num_nodes());
    for (std::size_t i = 0; i < graph.num_nodes(); ++i) {
        if (data_costs.col(i).empty()) continue;

        std::vector<std::size_t> adj_nodes = graph.get_adj_nodes(i);
        for (std::size_t j = 0; j < adj_nodes.size(); ++j) {
            std::size_t adj_node_id = adj_nodes[j];
            if (data_costs.col(adj_node_id).empty()) continue;

            /* Uni directional */
            if (i < adj_node_id) {
                mgraph.add_edge(i, adj_node_id, 1.0f);
            }
        }
    }
    mgraph.update_components();

    std::cout << mgraph.num_nodes() << " nodes, " << mgraph.num_edges() << " edges, " << mgraph.num_components() << "components" << std::endl;

    mapmap::LabelSet<cost_t, simd_w> label_set(graph.num_nodes(), false);
    for (std::size_t i = 0; i < data_costs.cols(); ++i) {
        DataCosts::Column const & data_costs_for_node = data_costs.col(i);

        std::vector<mapmap::_iv_st<cost_t, simd_w> > labels;
        if (data_costs_for_node.empty()) {
            labels.push_back(0);
        } else {
            labels.resize(data_costs_for_node.size());
            for(std::size_t j = 0; j < data_costs_for_node.size(); ++j) {
                labels[j] = data_costs_for_node[j].first + 1;
            }
        }

        label_set.set_label_set_for_node(i, labels);
    }

    std::vector<unary_t> unaries;
    unaries.reserve(data_costs.cols());
    pairwise_t pairwise(1.0f);
    for (std::size_t i = 0; i < data_costs.cols(); ++i) {
        DataCosts::Column const & data_costs_for_node = data_costs.col(i);

        std::vector<mapmap::_s_t<cost_t, simd_w> > costs;
        if (data_costs_for_node.empty()) {
            costs.push_back(1.0f);
        } else {
            costs.resize(data_costs_for_node.size());
            for(std::size_t j = 0; j < data_costs_for_node.size(); ++j) {
                float cost = data_costs_for_node[j].second;
                costs[j] = cost;
            }

        }

        unaries.emplace_back(i, &label_set);
        unaries.back().set_costs(costs);
    }

    mapmap::StopWhenReturnsDiminish<cost_t, simd_w> terminate(5, 0.01);
    std::vector<mapmap::_iv_st<cost_t, simd_w> > solution;

    auto display = [](const mapmap::luint_t time_ms,
            const mapmap::_iv_st<cost_t, simd_w> objective) {
        std::cout << "\t\t" << time_ms / 1000 << "\t" << objective << std::endl;
    };

    /* Create mapMAP solver object. */
    mapmap::mapMAP<cost_t, simd_w> solver;
    solver.set_graph(&mgraph);
    solver.set_label_set(&label_set);
    for(std::size_t i = 0; i < graph.num_nodes(); ++i)
        solver.set_unary(i, &unaries[i]);
    solver.set_pairwise(&pairwise);
    solver.set_logging_callback(display);
    solver.set_termination_criterion(&terminate);

    /* Pass configuration arguments (optional) for solve. */
    mapmap::mapMAP_control ctr;
    ctr.use_multilevel = true;
    ctr.use_spanning_tree = true;
    ctr.use_acyclic = true;
    ctr.spanning_tree_multilevel_after_n_iterations = 5;
    ctr.force_acyclic = true;
    ctr.min_acyclic_iterations = 5;
    ctr.relax_acyclic_maximal = true;
    ctr.tree_algorithm = mapmap::LOCK_FREE_TREE_SAMPLER;

    /* Set false for non-deterministic (but faster) mapMAP execution. */
    ctr.sample_deterministic = true;
    ctr.initial_seed = 548923723;

    std::cout << "\tOptimizing:\n\t\tTime[s]\tEnergy" << std::endl;
    solver.optimize(solution, ctr);

    /* Label 0 is undefined. */
    std::size_t num_labels = data_costs.rows() + 1;
    std::size_t undefined = 0;
    /* Extract resulting labeling from solver. */
    for (std::size_t i = 0; i < graph.num_nodes(); ++i) {
        int label = label_set.label_from_offset(i, solution[i]);
        if (label < 0 || num_labels <= static_cast<std::size_t>(label)) {
            throw std::runtime_error("Incorrect labeling");
        }
        if (label == 0) undefined += 1;
        else {
            graph.set_label(i, static_cast<std::size_t>(label));
            cell_labels[i] = label - 1;
        }
    }
    std::cout << std::endl;
    std::cout << '\t' << undefined << " cells have not been seen" << std::endl;
}

#ifdef WITH_GCO_OPTIMIZER
void TDOM::GraphOptimization(const std::vector<float> & image_qualities,
                            const std::vector<cv::Mat> & bitmaps, const MatXf & zBuffer, 
                            const std::unordered_map<image_pair_t, float> & image_pairs,
                            const std::vector<std::vector<int> > & super_ids_per_image,
                            const std::vector<std::unordered_set<uint32_t> >& neighbors_per_super,
                            const std::vector<std::vector<uint32_t> >& super_visibilities,
                            std::unordered_map<uint32_t, std::vector<uint32_t> > & super_idx_to_pixels,
                            std::unordered_map<uint32_t, int> & cell_labels) {
    const int num_image = model_.images.size();
    const int num_super_pixel = super_visibilities.size();
    const int sub_width = zBuffer.GetWidth();
    const int sub_height = zBuffer.GetHeight();
    const float gsd = options_.gsd;

    Timer timer;
    timer.Start();

    int num_image_pair = 0;
    std::unordered_map<image_pair_t, float> image_pair_scores;

    std::vector<image_pair_t> image_pair_ids;
    image_pair_ids.reserve(image_pairs.size());
    for (auto image_pair : image_pairs) {
        image_pair_ids.push_back(image_pair.first);
        image_pair_scores[image_pair.first] = 1.0f;
    }
    std::sort(image_pair_ids.begin(), image_pair_ids.end());

    std::cout << "Number of image pair: " << image_pair_ids.size() << std::endl;

#pragma omp parallel for schedule(dynamic)
     for (int k = 0; k < image_pair_ids.size(); ++k) {
        auto image_pair_id = image_pair_ids.at(k);
        image_t image_id1;
        image_t image_id2;
        utility::PairIdToImagePair(image_pair_id, &image_id1, &image_id2);

        const mvs::Image& image1 = model_.images.at(image_id1);
        const mvs::Image& image2 = model_.images.at(image_id2);
        const int image_width1 = image1.GetWidth();
        const int image_height1 = image1.GetHeight();
        const int image_width2 = image2.GetWidth();
        const int image_height2 = image2.GetHeight();
        Eigen::RowMatrix3x4f P1(image1.GetP());
        Eigen::RowMatrix3x4f P2(image2.GetP());

        const cv::Mat & bitmap1 = bitmaps.at(image_id1);
        const cv::Mat & bitmap2 = bitmaps.at(image_id2);

        auto super_ids1 = super_ids_per_image.at(image_id1);
        auto super_ids2 = super_ids_per_image.at(image_id2);

        Histogram hist1_Y(0.0f, 1.0f, 50);
        Histogram hist1_Cb(0.0f, 1.0f, 50);
        Histogram hist1_Cr(0.0f, 1.0f, 50);
        Histogram hist2_Y(0.0f, 1.0f, 50);
        Histogram hist2_Cb(0.0f, 1.0f, 50);
        Histogram hist2_Cr(0.0f, 1.0f, 50);

        std::vector<int> inter_super_ids;
        std::set_intersection(super_ids1.begin(), super_ids1.end(), super_ids2.begin(), super_ids2.end(), 
                              std::back_inserter(inter_super_ids));

        std::unordered_set<int> unique_pixels;
        for (auto super_id : inter_super_ids) {
            auto & pixels_per_super = super_idx_to_pixels.at(super_id);
            for (auto pixel : pixels_per_super) {
                unique_pixels.insert(pixel);
            }
        }
        std::vector<int> pixels;
        pixels.reserve(unique_pixels.size());
        pixels.insert(pixels.end(), unique_pixels.begin(), unique_pixels.end());

        // float area = pixels.size() * 1.0f / (sub_width * sub_height);
        // std::cout << StringPrintf("%d %d %f\n", pixels.size(), sub_width * sub_height, area);
        // if (area < 0.03) continue;

        for (auto pixel_id : pixels) {
            int r = pixel_id / sub_width;
            int c = pixel_id % sub_width;

            float z = zBuffer.Get(r, c);
            if (z >= FLT_MAX) {
                continue;
            }
            Eigen::Vector3d Xd;
            Xd[0] = (c + left_) * gsd + x_min_;
            Xd[1] = (r + top_) * gsd + y_min_;
            Xd[2] = z + z_min_;
            Eigen::Vector3f X = (inv_trans_to_utm_ * Xd.homogeneous()).head<3>().cast<float>();

            Eigen::Vector3f proj1 = P1 * X.homogeneous();
            int u1 = proj1.x() / proj1.z();
            int v1 = proj1.y() / proj1.z();
            if (u1 < 0 || u1 >= image_width1 || v1 < 0 || v1 >= image_height1) {
                continue;
            }
            cv::Vec3b color1 = bitmap1.at<cv::Vec3b>(v1, u1);
            float V1[3];
            V1[0] = color1[2] * INV_COLOR_NORM;
            V1[1] = color1[1] * INV_COLOR_NORM;
            V1[2] = color1[0] * INV_COLOR_NORM;
            ColorRGBToYCbCr(V1);

            Eigen::Vector3f proj2 = P2 * X.homogeneous();
            int u2 = proj2.x() / proj2.z();
            int v2 = proj2.y() / proj2.z();
            if (u2 < 0 || u2 >= image_width2 || v2 < 0 || v2 >= image_height2) {
                continue;
            }
            cv::Vec3b color2 = bitmap2.at<cv::Vec3b>(v2, u2);
            float V2[3];
            V2[0] = color2[2] * INV_COLOR_NORM;
            V2[1] = color2[1] * INV_COLOR_NORM;
            V2[2] = color2[0] * INV_COLOR_NORM;
            ColorRGBToYCbCr(V2);

            hist1_Y.add_value(V1[0]);
            hist1_Cb.add_value(V1[1]);
            hist1_Cr.add_value(V1[2]);
            hist2_Y.add_value(V2[0]);
            hist2_Cb.add_value(V2[1]);
            hist2_Cr.add_value(V2[2]);
        }
        
        int num_residual = 0;
        float his1, his2, error_Y(0.f), error_Cb(0.f), error_Cr(0.f);
        float bi = 0.01f;
        while (bi < 1.0f) {

            his1 = hist1_Y.get_approx_percentile(bi);
            his2 = hist2_Y.get_approx_percentile(bi);
            error_Y += std::fabs(his1 - his2);

            his1 = hist1_Cb.get_approx_percentile(bi);
            his2 = hist2_Cb.get_approx_percentile(bi);
            error_Cb += std::fabs(his1 - his2);

            his1 = hist1_Cr.get_approx_percentile(bi);
            his2 = hist2_Cr.get_approx_percentile(bi);
            error_Cr += std::fabs(his1 - his2);

            bi += 0.02f;
            num_residual++;
        }
        num_image_pair++;
        error_Y /= num_residual;
        error_Cb /= num_residual;
        error_Cr /= num_residual;
        image_pair_scores[image_pair_id] = 0.6 * error_Y + 0.2 * error_Cb + 0.2 * error_Cr;
        std::cout << StringPrintf("\rProcess image pair: [%6d, %6d] %d/%d, smooth cost = %f", image_id1, image_id2, 
                                  num_image_pair, image_pair_ids.size(), image_pair_scores[image_pair_id]);
    }
    std::cout << std::endl;

    try {
    GCoptimizationGeneralGraph gco(static_cast<GCoptimization::SiteID>(num_super_pixel), num_image + 1);

    // Set DataCost.
    std::vector<std::vector<GCoptimization::SparseDataCost> > sparse_data_costs;
    sparse_data_costs.resize(num_image + 1);

    std::vector<int> num_cell_of_image(num_image, 0);
    for (int super_id = 1; super_id < num_super_pixel; ++super_id) {
        auto & image_ids = super_visibilities.at(super_id);
        for (auto image_id : image_ids) {
            num_cell_of_image[image_id]++;
        }
    }
    for (int label = 0; label < num_image; ++label) {
        sparse_data_costs.at(label + 1).reserve(num_cell_of_image[label]);
    }

    std::cout << "Construct Sparse Data Cost." << std::endl;

    float min_data_cost = FLT_MAX;
    float max_data_cost = 0;
    for (int i = 0; i < num_image; ++i) {
        min_data_cost = std::min(min_data_cost, image_qualities.at(i));
        max_data_cost = std::max(max_data_cost, image_qualities.at(i));
    }
    float inv_norm_data_cost = 1.0f / (max_data_cost - min_data_cost);
    std::cout << StringPrintf("Normalize data cost: %f %f %f\n", min_data_cost, max_data_cost, inv_norm_data_cost);

// #pragma omp parallel for schedule(dynamic)
    int num_node_of_opt = 0;
    for (int super_id = 1; super_id < num_super_pixel; ++super_id) {
        if (super_idx_to_pixels.at(super_id).empty()) {
            continue;
        }
        auto & image_ids = super_visibilities.at(super_id);
        std::vector<std::pair<uint32_t, float> > image_scores;
        image_scores.reserve(image_ids.size());
        for (auto & image_id : image_ids) {
            image_scores.emplace_back(image_id, image_qualities.at(image_id));
        }

        // int max_num_label = std::min((int)image_ids.size(), 5);
        int max_num_label = image_ids.size();
        std::nth_element(image_scores.begin(), image_scores.begin() + max_num_label, image_scores.end(),
            [](const std::pair<uint32_t, float> &s1, const std::pair<uint32_t, float> &s2) {
                return s1.second < s2.second;
            });

        std::vector<std::pair<uint32_t, float> > image_scores_sorted;
        image_scores_sorted.resize(max_num_label);
        for (int k = 0; k < max_num_label; ++k) {
            image_scores_sorted[k] = image_scores.at(k);
        }

        std::sort(image_scores_sorted.begin(), image_scores_sorted.end(), 
            [&](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
                return a.first < b.first;
            });
// #pragma omp critical
        for (int k = 0; k < max_num_label; ++k) {
            uint32_t image_id = image_scores_sorted.at(k).first;
            float norm_cost = (image_scores_sorted.at(k).second - min_data_cost) * inv_norm_data_cost;
            GCoptimization::SparseDataCost data_cost;
            data_cost.site = super_id;
            data_cost.cost = GCoptimization::EnergyTermType(GCO_MAX_DATATERM * norm_cost);
            sparse_data_costs[image_id + 1].push_back(data_cost);
        }
        num_node_of_opt++;
    }
    std::cout << StringPrintf("%d nodes need to be optimized!\n", num_node_of_opt);

    std::cout << "Setting Data Cost." << std::endl;

    for (int image_id = 0; image_id < num_image; ++image_id) {
        if (!sparse_data_costs.at(image_id + 1).empty()) {
            gco.setDataCost(image_id + 1, &sparse_data_costs.at(image_id + 1)[0], sparse_data_costs[image_id + 1].size());
        }
    }

    std::cout << "Setting Smooth Cost." << std::endl;

    float min_smooth_cost = FLT_MAX;
    float max_smooth_cost = 0;
    for (auto image_pair_id : image_pair_ids) {
        float score = image_pair_scores.at(image_pair_id);
        if (score < 1) {
            min_smooth_cost = std::min(min_smooth_cost, score);
            max_smooth_cost = std::max(max_smooth_cost, score);
        }
    }
    float inv_norm_smooth_cost = 1.0f / (max_smooth_cost - min_smooth_cost);
    std::cout << StringPrintf("Normalize smooth cost: %f %f %f\n", min_smooth_cost, max_smooth_cost, inv_norm_smooth_cost);

#if 1
    for (auto image_pair_id : image_pair_ids) {
        image_t image_id1;
        image_t image_id2;
        utility::PairIdToImagePair(image_pair_id, &image_id1, &image_id2);
        if (image_pair_scores.at(image_pair_id) < 1) {
            GCoptimization::EnergyTermType e = GCO_MAX_SMOOTHTERM * (image_pair_scores[image_pair_id] - min_smooth_cost) * inv_norm_smooth_cost;
            image_pair_scores[image_pair_id] = e;
        } else {
            image_pair_scores[image_pair_id] = GCO_MAX_SMOOTHTERM;
        }
    }
    FullGCOHelper helper_data(&zBuffer, &model_.images, &bitmaps, &neighbors_per_super, 
                              &super_visibilities, &super_idx_to_pixels, &image_pair_scores);
    gco.setSmoothCost(&FullGCOHelper::SmoothFunc, &helper_data);
#else
    for (auto image_pair_id : image_pair_ids) {
        image_t image_id1;
        image_t image_id2;
        utility::PairIdToImagePair(image_pair_id, &image_id1, &image_id2);
        if (image_pair_scores.at(image_pair_id) < 1) {
            GCoptimization::EnergyTermType e = GCO_MAX_SMOOTHTERM * (image_pair_scores[image_pair_id] - min_smooth_cost) * inv_norm_smooth_cost;
            gco.setSmoothCost(image_id1 + 1, image_id2 + 1, e);
            gco.setSmoothCost(image_id2 + 1, image_id1 + 1, e);
        } else {
            gco.setSmoothCost(image_id1 + 1, image_id2 + 1, GCO_MAX_SMOOTHTERM);
            gco.setSmoothCost(image_id2 + 1, image_id1 + 1, GCO_MAX_SMOOTHTERM);
        }
    }
#endif
    std::cout << "Setting Neighbors." << std::endl;
    for (int super_id = 1; super_id < num_super_pixel; ++super_id) {
        if (super_visibilities.at(super_id).empty()) {
            continue;
        }
        for (auto neighbor_id : neighbors_per_super[super_id]) {
            if (super_id < neighbor_id) {
                gco.setNeighbors(super_id, neighbor_id);
            }
        }
        // std::cout << StringPrintf("\rProcess super id %d/%d", super_id, num_super_pixel) << std::flush;
    }
    std::cout << std::endl;
	std::cout << StringPrintf("\nBefore optimization energy is %lld",gco.compute_energy()) << std::flush;
    gco.swap(2);
	std::cout << StringPrintf("\nAfter optimization energy is %lld\n",gco.compute_energy()) << std::flush;

    cell_labels[0] = -1;
    for (int super_id = 1; super_id < num_super_pixel; ++super_id) {
        int label = gco.whatLabel(super_id);
        cell_labels[super_id] = label - 1;
    }
    }
    catch (GCException e) {
        e.Report();
    }
    timer.PrintMinutes();
}
#endif
}
} // namespace sensemap