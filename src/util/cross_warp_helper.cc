#include "cross_warp_helper.h"
#include "base/reconstruction.h"

using namespace sensemap;

PatchMatchCrossWarpHelper::PatchMatchCrossWarpHelper(const std::string &workspace_path, const std::unordered_set<image_t> &prior_depth_images, int num_images)
    : mvs::PatchMatchController(mvs::PatchMatchOptions{}, workspace_path),
      num_images_(num_images)
{
    mvs::Workspace::Options workspace_options;
    workspace_options.workspace_path = workspace_path;
    workspace_options.workspace_format = "panorama";
    workspace_.reset(new mvs::Workspace(workspace_options));

    Reconstruction reconstruction;
    reconstruction.ReadReconstruction(workspace_path);
    const auto &model = workspace_->GetModel();
    for (int image_idx = 0; image_idx < model.images.size(); ++image_idx) {
        image_t image_id = model.GetImageId(image_idx);
        if (prior_depth_images.count(image_id)) {
            prior_depth_images_.insert(image_idx);
        }
    }
}

std::vector<image_t> PatchMatchCrossWarpHelper::GetNeighboringImages(image_t ref_image_id) const {
    const auto &model = workspace_->GetModel();
    
    std::vector<image_t> src_image_ids;
    mvs::Problem problem;
    problem.ref_image_idx = model.GetImageIdx(ref_image_id);
    SelectSpatialNeighborViews(problem);
    if (problem.src_image_idxs.empty()) {
        return src_image_ids;
    }

    int src_idx = -1;
    for (auto idx : problem.src_image_idxs) {
        src_image_ids.emplace_back(model.GetImageId(idx));
    }

    return src_image_ids;
}

bool PatchMatchCrossWarpHelper::SelectSpatialNeighborViews(mvs::Problem &problem) const {
    const float optimal_angle = DegToRad(options_.optimal_angle);
    const float max_angle = DegToRad(70.0);
    const float min_angle = DegToRad(0.0);

    const mvs::Model &model = workspace_->GetModel();
    const mvs::Image &ref_image = model.images.at(problem.ref_image_idx);

    // std::cout << StringPrintf("ref_image_idx: %d, try to find spatial neighbor views", model.GetImageId(problem.ref_image_idx)) << std::endl;

    const float *K = ref_image.GetK();
    const float *R = ref_image.GetR();
    const float *t = ref_image.GetT();

    Eigen::Vector3f ref_C = -Eigen::Map<const Eigen::RowMatrix3f>(R).transpose()
                            * Eigen::Map<const Eigen::Vector3f>(t);
    Eigen::Vector3f ref_ray = Eigen::Map<const Eigen::RowMatrix3f>(R).row(2);

    size_t i, j;
    std::vector<float> dists(model.images.size(), 0.0f);
    for (i = 0; i < model.images.size(); ++i)
    {
        if (i == problem.ref_image_idx || !prior_depth_images_.count(i))
        {
            continue;
        }

        const mvs::Image &src_image = model.images.at(i);
        const float *src_R = src_image.GetR();
        const float *src_t = src_image.GetT();
        Eigen::Vector3f src_C = -Eigen::Map<const Eigen::RowMatrix3f>(src_R).transpose()
                                * Eigen::Map<const Eigen::Vector3f>(src_t);
        dists[i] = (src_C - ref_C).norm();
    }

    std::sort(dists.begin(), dists.end());
    const int nearest_neighbors = 20;
    int num_neighbor = 0;
    std::vector<float> k_nearest_dists;
    for (i = 0; (i < model.images.size()) && (num_neighbor < nearest_neighbors); ++i)
    {
        if (dists[i] < 1e-6)
        {
            continue;
        }
        num_neighbor++;
        k_nearest_dists.push_back(dists[i]);
    }
    if (k_nearest_dists.empty()) {
        return false;
    }
    size_t nth = k_nearest_dists.size() / 2;
    std::nth_element(k_nearest_dists.begin(), k_nearest_dists.begin() + nth, k_nearest_dists.end());
    float m_dist = k_nearest_dists.at(nth);

    struct ViewScore
    {
        uint32_t view_idx;
        float score;
    };
    std::vector<ViewScore> view_scores(model.images.size());
    for (i = 0; i < model.images.size(); ++i)
    {
        const mvs::Image &src_image = model.images.at(i);

        const float *src_R = src_image.GetR();
        const float *src_t = src_image.GetT();

        Eigen::Vector3f src_C = -Eigen::Map<const Eigen::RowMatrix3f>(src_R).transpose()
                                * Eigen::Map<const Eigen::Vector3f>(src_t);
        Eigen::Vector3f src_ray = Eigen::Map<const Eigen::RowMatrix3f>(src_R).row(2);

        const float cos_angle = ref_ray.dot(src_ray);
        const float angle = std::acos(cos_angle);
        const float wangle = std::max(angle / optimal_angle, 1.0f);
        const float dist = (src_C - ref_C).norm();
        const float wdist = std::max(dist / m_dist, 1.0f);

        ViewScore view_score;
        view_score.view_idx = i;
        if (angle > max_angle || angle < min_angle ||
            dist > 2 * m_dist || dist < 0.01 * m_dist || 
            !prior_depth_images_.count(i))
        {
            view_score.score = FLT_MAX;
        }
        else
        {
            view_score.score = wangle * wdist;
        }
        view_scores[i] = view_score;
    }

    std::sort(view_scores.begin(), view_scores.end(),
                [&](const ViewScore s1, const ViewScore s2)
                {
                    return s1.score < s2.score;
                });

    int min_num_src_images = std::min(view_scores.size(), (size_t)num_images_);

    if (min_num_src_images < options_.min_num_src_images)
    {
        std::cout << StringPrintf("Reference image %d has not enough images",
                                    problem.ref_image_idx)
                    << std::endl;
        return false;
    }

    // Update src_image_idxs.
    problem.src_image_idxs.clear();
    problem.src_image_scales.clear();
    for (i = 0, j = 0; (i < view_scores.size()) && (j < min_num_src_images); ++i)
    {
        ViewScore &view_score = view_scores.at(i);
        if (view_score.view_idx == problem.ref_image_idx ||
            view_score.score == FLT_MAX)
        {
            continue;
        }
        problem.src_image_idxs.push_back(view_score.view_idx);
        problem.src_image_scales.push_back(1.0f);
        j++;
    }

    return true;
}
