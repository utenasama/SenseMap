//Copyright (c) 2021, SenseTime Group.
//All rights reserved.
#include <map>
#include <iomanip>

#include "base/common.h"
#include "base/cost_functions.h"
#include "util/misc.h"
#include "util/ply.h"
#include "util/bitmap.h"
#include "util/mat.h"
#include "util/math.h"
#include "util/obj.h"
#include "util/threading.h"
#include "util/timer.h"
#include "util/proc.h"
#include "util/image_pair.h"
#include "util/histogram.h"
#include "util/color_space.h"
#include "mvs/workspace.h"
#include "graph/minimum_spanning_tree.h"

#include "base/version.h"
#include "../Configurator_yaml.h"

std::string configuration_file_path;

using namespace sensemap;

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

void GenerateRegionCorrespondences(const mvs::Model &model, 
                                   std::map<image_pair_t, std::vector<int> > &region_correspondence) {

    std::cout << "Generate Region Correspondence" << std::endl;

    const int num_images = model.images.size();

    std::map<image_pair_t, std::vector<int> > image_pairs;
    image_pairs.clear();
    for (size_t i = 0; i < model.points.size(); ++i) {
        const auto & point = model.points.at(i);
        for (size_t image_id1 : point.track) {
            for (size_t image_id2 : point.track) {
                if (image_id1 < image_id2) {
                    image_pair_t pair_id = utility::ImagePairToPairId(image_id1, image_id2);
                    image_pairs[pair_id].push_back(i);
                }
            }
        }
    }

    std::cout << "Graph Edge Compression" << std::endl;

    std::vector<std::pair<image_pair_t, int> > sorted_image_pairs;
    sorted_image_pairs.reserve(image_pairs.size());

    MinimumSpanningTree<image_t, int> mst_extractor;

    for (auto image_pair : image_pairs) {
        image_t image_id1;
        image_t image_id2;
        utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
        
        mst_extractor.AddEdge(image_id1, image_id2, -(int)image_pair.second.size());
        sorted_image_pairs.emplace_back(image_pair.first, image_pair.second.size());
    }
    std::unordered_set<std::pair<image_t, image_t> > minimum_spanning_tree;
    mst_extractor.Extract(&minimum_spanning_tree);

    const int num_redundance_edges = 0.1 * (image_pairs.size() - num_images + 1);
    std::nth_element(sorted_image_pairs.begin(), sorted_image_pairs.begin() + num_redundance_edges + num_images - 1, 
                     sorted_image_pairs.end(), 
        [&](const std::pair<image_pair_t, int>& a, const std::pair<image_pair_t, int>& b) {
            return a.second > b.second;
        });

    for (auto image_pair : minimum_spanning_tree) {
        auto pair_id = utility::ImagePairToPairId(image_pair.first, image_pair.second);
        region_correspondence[pair_id] = image_pairs.at(pair_id);
    }
    int num_edge = 0;
    while(num_edge < num_redundance_edges) {
        auto pair_id = sorted_image_pairs.at(num_edge).first;
        if (region_correspondence.find(pair_id) == region_correspondence.end()) {
            region_correspondence[pair_id] = image_pairs.at(pair_id);
        }
        num_edge++;
    }
    std::cout << StringPrintf("%d edges are preserved, %d edges are pruned!\n",
        region_correspondence.size(), image_pairs.size() - region_correspondence.size());
}

void GraphOptimization(const std::string& dense_path, const mvs::Model &model, 
                       const std::map<image_pair_t, std::vector<int> > &region_correspondence) {
    const int num_image = model.images.size();
    std::vector<double> s_Y(num_image, 1.0f), s_Cb(num_image, 1.0f), s_Cr(num_image, 1.0f);
    std::vector<double> o_Y(num_image, 0.0f), o_Cb(num_image, 1.0f), o_Cr(num_image, 1.0f);

    ceres::Problem problem_Y, problem_Cb, problem_Cr;
    ceres::LossFunction *loss_function_Y = new ceres::SoftLOneLoss(1.0);
    ceres::LossFunction *loss_function_Cb = new ceres::SoftLOneLoss(1.0);
    ceres::LossFunction *loss_function_Cr = new ceres::SoftLOneLoss(1.0);

    int num_image_pair = 0;

    std::vector<image_pair_t> image_pair_ids;
    image_pair_ids.reserve(region_correspondence.size());
    std::vector<std::vector<int> > region_points;
    region_points.reserve(region_correspondence.size());
    for (auto & image_pair : region_correspondence) {
        image_pair_ids.push_back(image_pair.first);
        region_points.push_back(image_pair.second);
    }

#pragma omp parallel for schedule(dynamic)
    // for (auto & image_pair : region_correspondence) {
    for (int k = 0; k < image_pair_ids.size(); ++k) {
        auto image_pair_id = image_pair_ids.at(k);
        image_t image_id1;
        image_t image_id2;
        utility::PairIdToImagePair(image_pair_id, &image_id1, &image_id2);

        Bitmap bitmap1, bitmap2;
#pragma omp critical
        {
            if (!cache.Exists(image_id1)) {
                cache.Add(image_id1, model.images.at(image_id1).GetPath());
            }
            bitmap1 = cache.Get(image_id1);

            if (!cache.Exists(image_id2)) {
                cache.Add(image_id2, model.images.at(image_id2).GetPath());
            }
            bitmap2 = cache.Get(image_id2);
        }

        Eigen::RowMatrix3x4f P1(model.images.at(image_id1).GetP());
        Eigen::RowMatrix3x4f P2(model.images.at(image_id2).GetP());
        Eigen::Vector4i region1, region2;
        region1[0] = region2[0] = std::numeric_limits<int>::max();
        region1[1] = region2[1] = std::numeric_limits<int>::max();
        region1[2] = region2[2] = std::numeric_limits<int>::lowest();
        region1[3] = region2[3] = std::numeric_limits<int>::lowest();

        // Find common region.
        for (auto & point_id : region_points.at(k)) {
            Eigen::Vector3f X(&model.points.at(point_id).x);
            Eigen::Vector3f proj1 = P1 * X.homogeneous();
            int u1 = proj1.x() / proj1.z();
            int v1 = proj1.y() / proj1.z();
            region1[0] = std::min(region1[0], u1);
            region1[1] = std::min(region1[1], v1);
            region1[2] = std::max(region1[2], u1);
            region1[3] = std::max(region1[3], v1);

            Eigen::Vector3f proj2 = P2 * X.homogeneous();
            int u2 = proj2.x() / proj2.z();
            int v2 = proj2.y() / proj2.z();
            region2[0] = std::min(region2[0], u2);
            region2[1] = std::min(region2[1], v2);
            region2[2] = std::max(region2[2], u2);
            region2[3] = std::max(region2[3], v2);
        }

        float overlap1 = ((region1[3] - region1[1]) * (region1[2] - region1[0])) * 1.0f /
                         (bitmap1.Width() * bitmap1.Height());
        float overlap2 = ((region2[3] - region2[1]) * (region2[2] - region2[0])) * 1.0f /
                         (bitmap2.Width() * bitmap2.Height());
        if (overlap1 < 0.2 || overlap2 < 0.2) {
            continue;
        }

        // Calculate Histograms.
        Histogram hist1_Y(0.0f, 1.0f, 10);
        Histogram hist1_Cb(0.0f, 1.0f, 10);
        Histogram hist1_Cr(0.0f, 1.0f, 10);
        for (int y = region1[1]; y <= region1[3]; ++y) {
            for (int x = region1[0]; x <= region1[2]; ++x) {
                BitmapColor<uint8_t> color;
                bitmap1.GetPixel(x, y, &color);

                float V[3];
                V[0] = color.r * INV_COLOR_NORM;
                V[1] = color.g * INV_COLOR_NORM;
                V[2] = color.b * INV_COLOR_NORM;
                ColorRGBToYCbCr(V);
                hist1_Y.add_value(V[0]);
                hist1_Cb.add_value(V[1]);
                hist1_Cr.add_value(V[2]);
            }
        }

        Histogram hist2_Y(0.0f, 1.0f, 10);
        Histogram hist2_Cb(0.0f, 1.0f, 10);
        Histogram hist2_Cr(0.0f, 1.0f, 10);
        for (int y = region2[1]; y <= region2[3]; ++y) {
            for (int x = region2[0]; x <= region2[2]; ++x) {
                BitmapColor<uint8_t> color;
                bitmap2.GetPixel(x, y, &color);
                float V[3];
                V[0] = color.r * INV_COLOR_NORM;
                V[1] = color.g * INV_COLOR_NORM;
                V[2] = color.b * INV_COLOR_NORM;
                ColorRGBToYCbCr(V);
                hist2_Y.add_value(V[0]);
                hist2_Cb.add_value(V[1]);
                hist2_Cr.add_value(V[2]);

            }
        }

#pragma omp critical
        {
        float bi = 0.05f;
        while (bi < 1.0f) {

            ceres::CostFunction *cost_function_Y = nullptr, *cost_function_Cb = nullptr, *cost_function_Cr = nullptr;

            cost_function_Y = ColorCorrectionCostFunction::Create(
                    hist1_Y.get_approx_percentile(bi),
                    hist2_Y.get_approx_percentile(bi));
            problem_Y.AddResidualBlock(cost_function_Y, loss_function_Y, &s_Y[image_id1],
                                       &s_Y[image_id2], &o_Y[image_id1], &o_Y[image_id2]);

            cost_function_Cb = ColorCorrectionCostFunction::Create(
                    hist1_Cb.get_approx_percentile(bi),
                    hist2_Cb.get_approx_percentile(bi));
            problem_Cb.AddResidualBlock(cost_function_Cb, loss_function_Cb, &s_Cb[image_id1],
                                        &s_Cb[image_id2], &o_Cb[image_id1], &o_Cb[image_id2]);

            cost_function_Cr = ColorCorrectionCostFunction::Create(
                    hist1_Cr.get_approx_percentile(bi),
                    hist2_Cr.get_approx_percentile(bi));
            problem_Cr.AddResidualBlock(cost_function_Cr, loss_function_Cr, &s_Cr[image_id1],
                                        &s_Cr[image_id2], &o_Cr[image_id1], &o_Cr[image_id2]);

            bi += 0.1f;
        }
        num_image_pair++;
        std::cout << StringPrintf("\rProcess image pair: [%d, %d] %d/%d", image_id1, image_id2, num_image_pair, region_correspondence.size()) << std::flush;
        }
    }
    std::cout << std::endl;

#pragma omp parallel for schedule(dynamic)
    for(std::size_t i = 0; i < num_image; ++i){
        if(problem_Y.HasParameterBlock(&s_Y[i])) {
            problem_Y.SetParameterLowerBound(&s_Y[i], 0, 1 - 0.4);
            problem_Y.SetParameterUpperBound(&s_Y[i], 0, 1 + 0.4);
            problem_Y.SetParameterLowerBound(&o_Y[i], 0, -30.0 / 255);
            problem_Y.SetParameterUpperBound(&o_Y[i], 0, 30.0 / 255);
        }
        if(problem_Cb.HasParameterBlock(&s_Cb[i])) {
            problem_Cb.SetParameterLowerBound(&s_Cb[i], 0, 1 - 0.2);
            problem_Cb.SetParameterUpperBound(&s_Cb[i], 0, 1 + 0.2);
            problem_Cb.SetParameterLowerBound(&o_Cb[i], 0, -5.0 / 255);
            problem_Cb.SetParameterUpperBound(&o_Cb[i], 0, 5.0 / 255);
        }
        if(problem_Cr.HasParameterBlock(&s_Cr[i])) {
            problem_Cr.SetParameterLowerBound(&s_Cr[i], 0, 1 - 0.2);
            problem_Cr.SetParameterUpperBound(&s_Cr[i], 0, 1 + 0.2);
            problem_Cr.SetParameterLowerBound(&o_Cr[i], 0, -5.0 / 255);
            problem_Cr.SetParameterUpperBound(&o_Cr[i], 0, 5.0 / 255);
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
}

int main(int argc, char *argv[]) {
    PrintHeading("SenseMap.  Copyright(c) 2022, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

	std::string workspace_path = param.GetArgument("workspace_path", "");
	std::string image_type = param.GetArgument("image_type", "perspective");

    for (size_t rec_idx = 0; ;rec_idx++) {
        auto reconstruction_path = JoinPaths(workspace_path, std::to_string(rec_idx));
        auto dense_reconstruction_path = JoinPaths(reconstruction_path, DENSE_DIR);
        auto sparse_path = JoinPaths(dense_reconstruction_path, SPARSE_DIR);
        auto undistort_image_path = JoinPaths(dense_reconstruction_path, IMAGES_DIR);
        if (!ExistsDir(reconstruction_path)) {
            break;
        }
        if (!ExistsDir(dense_reconstruction_path)) {
            std::cout << StringPrintf("dense path is not exist!\n");
            break;
        }

        Timer timer;
        timer.Start();

        mvs::Workspace::Options workspace_options;
        workspace_options.image_as_rgb = true;
        workspace_options.workspace_path = dense_reconstruction_path;
        workspace_options.workspace_format = image_type;
        workspace_options.image_path = undistort_image_path;

        std::unique_ptr<mvs::Workspace> workspace;
        workspace.reset(new mvs::Workspace(workspace_options));
        mvs::Model model = workspace->GetModel();

        std::map<image_pair_t, std::vector<int> > region_correspondence;
        GenerateRegionCorrespondences(model, region_correspondence);
        GraphOptimization(dense_reconstruction_path, model, region_correspondence);

        timer.PrintMinutes();
    }

    return 0;
}