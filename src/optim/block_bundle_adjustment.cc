// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include "optim/block_bundle_adjustment.h"

#include <iomanip>

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#include <iostream>

#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "base/pose.h"
#include "base/projection.h"
#include "util/misc.h"
#include "util/threading.h"
#include "util/timer.h"
#include "util/kmeans.h"
#include "optim/global_motions/utils.h"
#include "bundle_adjustment.h"
#include "estimators/triangulation.h"
#ifdef USE_OPENBLAS
#include "openblas/cblas.h"
#endif
namespace sensemap {

// debug
void DebugBlocks(
    const std::string dir,
    const std::unordered_map<block_t, std::shared_ptr<Block>> &blocks) {
    for (const auto &block : blocks) {
        FILE *fp = fopen((dir + "_block_map_" + std::to_string(block.first) + ".obj").c_str(), "w+");
        
        const auto reconstruction = block.second->reconstruction;
        auto bk_mappoints = reconstruction->MapPoints();
        auto bk_constant_mappoints = block.second->constant_mappoints;
        auto bk_images = reconstruction->Images();

        int r = random() % 255;
        int g = random() % 255;
        int b = random() % 255;

        for (const auto &map : bk_mappoints) {
            const auto &mappt = map.second.XYZ();
            if (bk_constant_mappoints.count(map.first)) 
                fprintf(fp, "v %f %f %f 255 255 255\n", mappt[0], mappt[1], mappt[2]);
            else
                fprintf(fp, "v %f %f %f %d %d %d\n", mappt[0], mappt[1], mappt[2], r, g, b);
        }
        fclose(fp);

        fp = fopen((dir + "_block_camera_" + std::to_string(block.first) + ".obj").c_str(), "w+");
        r = random() % 255;
        g = random() % 255;
        b = random() % 255;

        for (const auto &img : bk_images) {
            const auto &img_r = img.second.RotationMatrix();
            const auto &img_t = img.second.Tvec();

            const auto img_position = -img_r.transpose() * img_t;

            if (block.second->common_images.count(img.first) == 0)
            {
                fprintf(fp, "v %f %f %f %d %d %d\n", img_position[0], img_position[1], img_position[2], r, g, b);
            } else {
                fprintf(fp, "v %f %f %f 255 255 255\n", img_position[0], img_position[1], img_position[2]);
            }
        }
        fclose(fp);


        fp = fopen((dir + "_block_camera_" + std::to_string(block.first) + ".txt").c_str(), "w+");
        std::vector<image_t> imgsets;
        for (const auto &img : bk_images) imgsets.emplace_back(img.first);
        std::sort(imgsets.begin(), imgsets.end());
        for (int i = 0; i < imgsets.size(); ++ i) {
            fprintf(fp, "%d\n", imgsets[i]);
        }
        fclose(fp);

    }
} 


// debug
void DebugReconstruction(
    const std::string dir,
    const Reconstruction *reconstruction) {

    FILE *fp = fopen((dir + "_map.obj").c_str(), "w+");

    auto bk_mappoints = reconstruction->MapPoints();
    auto bk_images = reconstruction->Images();

    int r = random() % 255;
    int g = random() % 255;
    int b = random() % 255;

    for (const auto &map : bk_mappoints) {
        const auto &mappt = map.second.XYZ();
        fprintf(fp, "v %f %f %f %d %d %d\n", mappt[0], mappt[1], mappt[2], r, g, b);
    }
    fclose(fp);


    fp = fopen((dir + "_camera.obj").c_str(), "w+");
    r = random() % 255;
    g = random() % 255;
    b = random() % 255;

    for (const auto &img : bk_images) {
        if (!img.second.IsRegistered()) continue;

        const auto &img_r = img.second.RotationMatrix();
        const auto &img_t = img.second.Tvec();

        const auto img_position = -img_r.transpose() * img_t;

        fprintf(fp, "v %f %f %f %d %d %d\n", img_position[0], img_position[1], img_position[2], r, g, b);
    }
    fclose(fp);
}


//
BlockBundleAdjuster::BlockBundleAdjuster(
    const BundleAdjustmentOptions& options, const BundleAdjustmentConfig& config) 
    : options_(options), config_(config)
{
    CHECK(options_.Check());

    block_options_ =
        BlockBundleAdjustmentOptions(config_.NumImages(), options.block_size, options.block_common_image_num,
                                     options.min_connected_points_for_common_images);
}

std::unordered_map<block_t, std::shared_ptr<Block>>
BlockBundleAdjuster::DivideBlocks(
    Reconstruction *reconstruction, 
    const BlockBundleAdjustmentOptions &option) const
{
    auto st = std::chrono::steady_clock::now();
    std::unordered_map<block_t, std::shared_ptr<Block>> blocks;

    auto images = config_.Images();
    auto variable_points = config_.VariablePoints();
    auto constant_points = config_.ConstantPoints();
    auto constant_cameras = config_.ConstantCameraIds();
    auto constant_poses = config_.ConstantPoses();
    auto constant_tvecs = config_.ConstantTVecs();
    auto all_cameras = reconstruction->Cameras();
    // const auto &correspondence = reconstruction->GetCorrespondenceGraph();
    
    auto AddImages = [](
        std::shared_ptr<Block> block, const image_t image,
        std::unordered_set<image_t> &constant_poses,
        std::unordered_map<image_t, std::vector<int>> &constant_tvecs) -> bool {
        block->config.AddImage(image);
        if (constant_poses.count(image)) block->config.SetConstantPose(image);
        if (constant_tvecs.count(image)) block->config.SetConstantTvec(image, constant_tvecs[image]);
        return true;
    };

    auto AddMappoints = [](
        std::shared_ptr<Block> block, 
        const std::unordered_set<mappoint_t> &mappoints,
        const std::unordered_set<mappoint_t> &constant_points,
        const std::unordered_set<mappoint_t> &variable_points) -> bool {
        for (const auto &mappt : mappoints) {
            if (constant_points.count(mappt)) block->config.AddConstantPoint(mappt);
            if (variable_points.count(mappt)) block->config.AddVariablePoint(mappt);
        }
        return true;
    };

    // image2mappoints
    const auto &all_mappoints = reconstruction->MapPointIds();
    typedef std::unordered_map<image_t, std::unordered_set<mappoint_t>> IMAGE2MAP;
    typedef std::unordered_map<image_pair_t, int> CORRESMAP;

    int threads_num = 10;

    std::vector<mappoint_t> all_mappoints_vec;
    all_mappoints_vec.assign(all_mappoints.begin(), all_mappoints.end());

    std::cout<<"Covisibility computation thread: "<< threads_num << std::endl;
    std::vector<IMAGE2MAP> omp_image2mappoints(threads_num);
    std::vector<CORRESMAP> omp_correspondence_mapper(threads_num);
    std::vector<std::vector<mappoint_t>> mappoint_threads(threads_num);

    IMAGE2MAP image2mappoints;
    CORRESMAP correspondence_mapper;

    #pragma omp parallel for schedule(dynamic) num_threads(threads_num)
    for (int p = 0; p < all_mappoints_vec.size(); ++ p) {
        int tid = omp_get_thread_num();

        auto &tmp_image2mappoints = omp_image2mappoints[tid];
        auto &tmp_correspondence_mapper = omp_correspondence_mapper[tid];

        const auto map_id = all_mappoints_vec[p];
        class MapPoint& mappoint = reconstruction->MapPoint(map_id);
        class Track& track = mappoint.Track();
        int track_length = track.Length();
        const auto &track_eles = track.Elements();

        for (size_t i = 0; i < track_length; ++i) {
            const TrackElement& track_el = track_eles[i];
            tmp_image2mappoints[track_el.image_id].insert(map_id);
         
        }
        if(track_length > 200){
            continue;
        }
        for (size_t i = 0; i < track_length; ++ i) {
            for (size_t j = i + 1; j < track_length; ++ j) {
                image_t s = track_eles[i].image_id;
                image_t e = track_eles[j].image_id;

                const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(s, e);

                int cnt = 0;
                if (tmp_correspondence_mapper.count(pair_id)) cnt = tmp_correspondence_mapper[pair_id];
                
                tmp_correspondence_mapper[pair_id] = cnt + 1;
            }
        }
    }

    for (int i = 0; i < threads_num; ++ i) {
        const auto &tmp_image2mappoints = omp_image2mappoints[i];
        const auto &tmp_correspondence_mapper = omp_correspondence_mapper[i];

        for (const auto item : tmp_image2mappoints) {
            image2mappoints[item.first].insert(item.second.begin(), item.second.end());
        }

        for (const auto item : tmp_correspondence_mapper) {
            int cnt = 0;
            if (correspondence_mapper.count(item.first)) cnt = correspondence_mapper[item.first];
            correspondence_mapper[item.first] = cnt + item.second;
        }
    }

    auto ed = std::chrono::steady_clock::now();
    auto t1 = std::chrono::duration<float, std::milli>(ed - st).count() * 0.001f;

    st = std::chrono::steady_clock::now();
    KMeans kmeans;
    size_t cluster_num = (images.size() - 1) / option.max_block_images + 1;
    kmeans.SetK(cluster_num);
    kmeans.SetFixedSize(option.block_images_num);
    
    for (const auto &img : images) {
        const auto &image = reconstruction->Image(img);
        const auto &r = image.RotationMatrix();
        const auto &t = image.Tvec();
        const auto &location = -r.transpose() * t;

        Tuple tuple;
        tuple.id = img;
        tuple.location = location;
        kmeans.mv_pntcloud.emplace_back(tuple);
    }
    bool kmeans_res = kmeans.SameSizeCluster();
    if (!kmeans_res) {
        std::unordered_map<block_t, std::shared_ptr<Block>> empty_blocks;
        return empty_blocks;
    }

    // divide blocks
    block_t block_count = 0;
    std::unordered_set<image_t> valid_images;
    for (int i = 0; i < kmeans.mv_center.size(); ++ i) {
        // create new block
        std::shared_ptr<Block> block = std::make_shared<Block>(block_count);
        ++ block_count;
        for (const auto &camera : all_cameras) {
            block->config.SetConstantCamera(camera.first);
        }

        // find max groups
        const auto &tuples = kmeans.m_grp_pntcloud[i];
        DisjointSet jset(tuples.size());
        for (int ti = 0; ti < tuples.size(); ++ ti) {
            for (int tj = ti + 1; tj < tuples.size(); ++ tj) {
                const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(tuples[ti].id, tuples[tj].id);
                if (correspondence_mapper.count(pair_id)) {
                    jset.Merge(ti, tj);
                }
            }
        }
        const auto &max_group_idxs = jset.MaxComponent();
        std::unordered_set<mappoint_t> visited_mappoints;
        for (const auto &idx : max_group_idxs) {
            image_t cur_image = tuples[idx].id;

            // 1. add image / pose / tvec
            AddImages(block, cur_image, constant_poses, constant_tvecs);

            // 2. add map points: constant points / variable points
            auto new_mappoints = image2mappoints[cur_image];
            AddMappoints(block, new_mappoints, constant_points, variable_points);

            valid_images.insert(cur_image);
            visited_mappoints.insert(new_mappoints.begin(), new_mappoints.end());
            block->image_sum_location += tuples[idx].location;
        }
        block->mappoints = visited_mappoints;
        blocks[block->id] = block;
    }

    // remain images
    std::unordered_set<image_t> remain_images;
    for (const auto &img : images) {
        if (valid_images.count(img)) continue;
        remain_images.insert(img);
    }
    int prev_remain_imgsize = remain_images.size();
    while (!remain_images.empty()) {
        for (const auto img : remain_images) {

            block_t target_block_id = 0;
            size_t max_covisible = 0;
            for (const auto &block : blocks) {
                const auto &block_imgs = block.second->config.Images();

                size_t covisible_cnt = 0;
                for (const auto &blk_img : block_imgs) {
                    const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(img, blk_img);

                    if (correspondence_mapper.count(pair_id) == 0) continue;

                    auto covis = correspondence_mapper[pair_id];
                    if (covisible_cnt < covis) covisible_cnt = covis;
                }

                if (covisible_cnt > max_covisible) {
                    max_covisible = covisible_cnt;
                    target_block_id = block.first;
                }
            }

	        if (max_covisible == 0) continue;
            auto block = blocks[target_block_id];

            const auto &mappoints = image2mappoints[img];
            // add to block
            // 1. add image / pose / tvec
            AddImages(block, img, constant_poses, constant_tvecs);
        
            // 2. add map points: constant points / variable points
            AddMappoints(block, mappoints, constant_points, variable_points);

            block->mappoints.insert(mappoints.begin(), mappoints.end());

            valid_images.insert(img);
        }

        remain_images.clear();
        for (const auto &img : images) {
            if (valid_images.count(img)) continue;
            remain_images.insert(img);
        }

        if (remain_images.size() == prev_remain_imgsize) {
            std::cout << "BlockBA Err : can not find block for images: " << std::endl;
            for (auto img : remain_images) {
                std::cout << img << ' ';
            }
            std::cout << std::endl;

            std::unordered_map<block_t, std::shared_ptr<Block>> empty_blocks;
            //break;
            return empty_blocks;
        }

        prev_remain_imgsize = remain_images.size();
    }
    ed = std::chrono::steady_clock::now();
    auto t2 = std::chrono::duration<float, std::milli>(ed - st).count() * 0.001f;

    st = std::chrono::steady_clock::now();
    // Add common images
    const int common_images_thres = option.common_images_num;
    typedef struct Node {
        image_t image_id;
        int votes;
        bool operator < (const struct Node &other) const {
            return votes > other.votes;
        }
        Node(const image_t in_image_id, const int in_votes) {
            image_id = in_image_id;
            votes = in_votes;
        }
    } Node;
    std::unordered_map<block_t, std::unordered_set<image_t>> common_img_mapper;
    for (auto &dst_block : blocks) {
        auto &dst_config = dst_block.second->config;
        const auto &dst_block_images = dst_config.Images();
        block_t dst_block_id = dst_block.first;

        for (auto &src_block : blocks) {
            if (src_block.first == dst_block.first) continue;

            block_t src_block_id = src_block.first;
            const auto &src_config = src_block.second->config;
            std::vector<Node> votes;
            const auto &src_block_images = src_config.Images();

            for (const auto &src_image : src_block_images) {
                int vote = 0;
                for (const auto &dst_img : dst_block_images) {
        	    const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(src_image, dst_img);

                    if (correspondence_mapper.count(pair_id) == 0) continue;

                    auto covisible = correspondence_mapper[pair_id];

                    if (vote < covisible) vote = covisible;
                }

                if (vote > option.min_connected_points_for_common_images) votes.emplace_back(Node(src_image, vote));
            }

            std::sort(votes.begin(), votes.end());
            for (int c = 0; c < votes.size() && c < common_images_thres; ++ c) {
                image_t cur_image = votes[c].image_id;

                common_img_mapper[dst_block_id].insert(cur_image);
                common_img_mapper[src_block_id].insert(cur_image);
            }
        }
    }
    ed = std::chrono::steady_clock::now();
    auto t3 = std::chrono::duration<float, std::milli>(ed - st).count() * 0.001f;

    st = std::chrono::steady_clock::now();
    for (const auto &common_mapper : common_img_mapper) {
        const auto &block_id = common_mapper.first;
        const auto &common_imgs = common_mapper.second;
        auto &block = blocks[block_id];
        for (const auto &cur_image : common_imgs) {
            block->common_images.insert(cur_image);
            // 1. add image / pose / tvec
            AddImages(block, cur_image, constant_poses, constant_tvecs);
            
            // 2. add map points: constant points / variable points
            auto visible_mappoints = image2mappoints[cur_image];
            AddMappoints(block, visible_mappoints, constant_points, variable_points);

            block->mappoints.insert(visible_mappoints.begin(), visible_mappoints.end());
        }
    }
    ed = std::chrono::steady_clock::now();
    auto t4 = std::chrono::duration<float, std::milli>(ed - st).count() * 0.001f;

    // copy reconstruction
    st = std::chrono::steady_clock::now();
    std::unordered_map<block_t, std::unordered_set<mappoint_t>> constant_mappts;
    for (const auto blk : blocks) {
        auto block_id = blk.first;
        auto block = blk.second;
        const auto &blk_pts = block->mappoints;
        auto &constant_pts = constant_mappts[block_id];

        for (const auto pt : blk_pts) {
            const auto &map = reconstruction->MapPoint(pt);
            auto &track_eles = map.Track().Elements();

            bool keep_constant = false;
            for (int i = 0; i < track_eles.size(); ++ i) {
                image_t img = track_eles[i].image_id;
                if (block->config.HasImage(img) == false) {
                    keep_constant = true;
                    break;
                }
            }
            if (keep_constant) constant_pts.insert(pt);
        }
    }

    int omp_thres_num = blocks.size();
#pragma omp parallel for schedule(dynamic) num_threads(omp_thres_num)
    for (int i = 0; i < blocks.size(); ++ i) {
        auto &block = blocks[i];
        reconstruction->Copy(block->config.Images(), block->mappoints, block->reconstruction);
    }

    for (int i = 0; i < blocks.size(); ++ i) {
        auto &block = blocks[i];
        const auto &mappts = block->reconstruction->MapPoints();
        const auto &constant_pts = constant_mappts[block->id];

        block->mappoints.clear();
        for (const auto &pt : mappts)
        {
            block->mappoints.insert(pt.first);
            if (constant_pts.count(pt.first)) block->constant_mappoints.insert(pt.first);
        }
    }

    // fix block mappoints
    for (auto blk : blocks) {
        const auto &constant_pts = blk.second->constant_mappoints;
        for (const auto pt : constant_pts) {
            blk.second->config.RemoveVariablePoint(pt);
            blk.second->config.AddConstantPoint(pt);
        }
    }

    // fix block images
    if (!options_.use_prior_absolute_location || !reconstruction->b_aligned) {
        for (auto block : blocks) {
            const auto &images = block.second->config.Images();
            double max_distance = 0.0;
            image_t fst_img_id, snd_img_id;
            for (auto img1 : images) {
                for (auto img2 : images) {
                    if (img1 == img2) continue;

                    auto position1 = reconstruction->Image(img1).ProjectionCenter();
                    auto position2 = reconstruction->Image(img2).ProjectionCenter();
                    auto dist = (position1 - position2).norm();

                    if (dist > max_distance) {
                        max_distance = dist;
                        fst_img_id = img1;
                        snd_img_id = img2;
                    }
                }
            }
            
            block.second->config.RemoveConstantTvec(fst_img_id);
            block.second->config.SetConstantPose(fst_img_id);
            block.second->config.RemoveConstantTvec(snd_img_id);
            block.second->config.SetConstantPose(snd_img_id);
        }
    }
    ed = std::chrono::steady_clock::now();

    for (const auto &block : blocks) {
        const auto &images = block.second->config.Images();
        const auto &com_images = block.second->common_images;
        std::cout << "Block" << block.first << ' ' << images.size() << ' ' << com_images.size() << std::endl;
    }
    std::cout << "Config images: " << config_.NumImages() << ", Block images: " << valid_images.size() << ", initial time : " << t1 << std::endl;

    return blocks;
}

std::unordered_map<block_t, std::shared_ptr<Block>>
BlockBundleAdjuster::DivideBlocks2(
    Reconstruction *reconstruction, 
    const BlockBundleAdjustmentOptions &option) const
{
    auto st = std::chrono::steady_clock::now();
    std::unordered_map<block_t, std::shared_ptr<Block>> blocks;

    auto images = config_.Images();
    auto variable_points = config_.VariablePoints();
    auto constant_points = config_.ConstantPoints();
    auto constant_cameras = config_.ConstantCameraIds();
    auto constant_poses = config_.ConstantPoses();
    auto constant_tvecs = config_.ConstantTVecs();
    auto all_cameras = reconstruction->Cameras();
    const auto &correspondence = reconstruction->GetCorrespondenceGraph();
    
    auto AddImages = [](
        std::shared_ptr<Block> block, const image_t image,
        std::unordered_set<image_t> &constant_poses,
        std::unordered_map<image_t, std::vector<int>> &constant_tvecs) -> bool {
        block->config.AddImage(image);
        if (constant_poses.count(image)) block->config.SetConstantPose(image);
        if (constant_tvecs.count(image)) block->config.SetConstantTvec(image, constant_tvecs[image]);
        return true;
    };

    auto AddMappoints = [](
        std::shared_ptr<Block> block, 
        const std::unordered_set<mappoint_t> &mappoints,
        const std::unordered_set<mappoint_t> &constant_points,
        const std::unordered_set<mappoint_t> &variable_points) -> bool {
        for (const auto &mappt : mappoints) {
            if (constant_points.count(mappt)) block->config.AddConstantPoint(mappt);
            if (variable_points.count(mappt)) block->config.AddVariablePoint(mappt);
        }
        return true;
    };

    // image2mappoints
    const auto &all_mappoints = reconstruction->MapPointIds();
    std::unordered_map<image_t, std::unordered_set<mappoint_t>> image2mappoints;
    std::unordered_map<mappoint_t, std::unordered_set<image_t>> mappoint2image;
    for (const auto &map_id : all_mappoints) {
        class MapPoint& mappoint = reconstruction->MapPoint(map_id);
        class Track& track = mappoint.Track();

        for (size_t i = 0; i < track.Length(); ++i) {
            const TrackElement& track_el = track.Elements()[i];
            image2mappoints[track_el.image_id].insert(map_id);
            mappoint2image[map_id].insert(track_el.image_id);
        }
    }
    auto ed = std::chrono::steady_clock::now();
    auto t1 = std::chrono::duration<float, std::milli>(ed - st).count() * 0.001f;

    st = std::chrono::steady_clock::now();
    // divide blocks
    std::unordered_set<image_t> visited_images;
    block_t block_count = 0;

    for (const auto &image : images) {
        if (visited_images.count(image)) continue;

        std::shared_ptr<Block> block = std::make_shared<Block>(block_count);
        ++ block_count;

        // set cameras
        for (const auto &camera : all_cameras) {
            block->config.SetConstantCamera(camera.first);
        }

        std::queue<image_t> q;
        q.push(image);
        visited_images.insert(image);
        std::unordered_map<image_t, int> image_votes;
        std::unordered_set<mappoint_t> visited_mappoints;
        Eigen::Vector3d block_total_camposition = Eigen::Vector3d::Zero();
        while (!q.empty()) {
            image_t cur_image = q.front();
            q.pop();

            // 1. add image / pose / tvec
            AddImages(block, cur_image, constant_poses, constant_tvecs);

            // 2. add map points: constant points / variable points
            auto new_mappoints = image2mappoints[cur_image];
            AddMappoints(block, new_mappoints, constant_points, variable_points);
            for (auto iter = new_mappoints.begin(); iter != new_mappoints.end(); ) {
                if (visited_mappoints.count(*iter) != 0) iter = new_mappoints.erase(iter);
                else ++ iter;
            }
            visited_mappoints.insert(new_mappoints.begin(), new_mappoints.end());

            const auto &image = reconstruction->Image(cur_image);
            block_total_camposition += image.ProjectionCenter();
            Eigen::Vector3d block_center = block_total_camposition / block->config.NumImages();

            if (block->config.NumImages() >= option.block_images_num) break;

            // 3. find next image
            for (const auto &mappoint : new_mappoints) {
                const auto &imgs = mappoint2image[mappoint];
                for (const auto &img : imgs) {
                    if (visited_images.count(img)) continue;
                    if (image_votes.count(img) == 0) image_votes[img] = 0;
                    ++ image_votes[img];
                }
            }
            
            image_t next_image;
            double max_score = -1;
            for (const auto &image_vote : image_votes) {
                if (visited_images.count(image_vote.first)) continue;
                
                const auto &image = reconstruction->Image(image_vote.first);
                const auto &position = image.ProjectionCenter();
                double image_distance = (position - block_center).norm();
                double image_score = image_vote.second; //image_vote.second * 1.0 / std::sqrt(image_distance);
                if (max_score < image_score) {
                    max_score = image_score;
                    next_image = image_vote.first;
                }
            }
            if (max_score < 0) break;

            // 4. add next image
            q.push(next_image);
            visited_images.insert(next_image);
            image_votes.erase(next_image);
        }

        block->mappoints = visited_mappoints;
        blocks[block->id] = block;
    }
    ed = std::chrono::steady_clock::now();
    auto t2 = std::chrono::duration<float, std::milli>(ed - st).count() * 0.001f;

    st = std::chrono::steady_clock::now();
    // Add common images
    const int common_images_thres = option.common_images_num;
    typedef struct Node {
        image_t image_id;
        int votes;
        bool operator < (const struct Node &other) const {
            return votes > other.votes;
        }
        Node(const image_t in_image_id, const int in_votes) {
            image_id = in_image_id;
            votes = in_votes;
        }
    } Node;
    std::unordered_map<block_t, std::unordered_set<image_t>> common_img_mapper;
    for (auto &dst_block : blocks) {
        auto &dst_config = dst_block.second->config;
        const auto &dst_block_images = dst_config.Images();
        block_t dst_block_id = dst_block.first;

        for (auto &src_block : blocks) {
            if (src_block.first == dst_block.first) continue;

            block_t src_block_id = src_block.first;
            const auto &src_config = src_block.second->config;
            std::vector<Node> votes;
            const auto &src_block_images = src_config.Images();

            for (const auto &src_image : src_block_images) {
                int vote = 0;
                for (const auto &dst_img : dst_block_images) {
                    auto covisible = correspondence->NumCorrespondencesBetweenImages(src_image, dst_img);

                    if (vote < covisible) vote = covisible;
                }

                if (vote) votes.emplace_back(Node(src_image, vote));
                // const auto &src_mappoints = image2mappoints[src_image];

                // int vote = 0;
                // for (const auto &mappoint : src_mappoints) {
                //     if (dst_block.second.mappoints.count(mappoint)) ++ vote;
                // }
                // if (vote > 10) votes.emplace_back(Node(src_image, vote));
            }
            std::sort(votes.begin(), votes.end());
            for (int c = 0; c < votes.size() && c < common_images_thres; ++ c) {
                image_t cur_image = votes[c].image_id;

                common_img_mapper[dst_block_id].insert(cur_image);
                common_img_mapper[src_block_id].insert(cur_image);
            }
        }
    }
    ed = std::chrono::steady_clock::now();
    auto t3 = std::chrono::duration<float, std::milli>(ed - st).count() * 0.001f;

    st = std::chrono::steady_clock::now();
    for (const auto &common_mapper : common_img_mapper) {
        const auto &block_id = common_mapper.first;
        const auto &common_imgs = common_mapper.second;
        auto &block = blocks[block_id];
        for (const auto &cur_image : common_imgs) {
            block->common_images.insert(cur_image);
            // 1. add image / pose / tvec
            AddImages(block, cur_image, constant_poses, constant_tvecs);
            
            // 2. add map points: constant points / variable points
            auto visible_mappoints = image2mappoints[cur_image];
            AddMappoints(block, visible_mappoints, constant_points, variable_points);

            block->mappoints.insert(visible_mappoints.begin(), visible_mappoints.end());
        }
    }
    ed = std::chrono::steady_clock::now();
    auto t4 = std::chrono::duration<float, std::milli>(ed - st).count() * 0.001f;

    // copy reconstruction
    st = std::chrono::steady_clock::now();
    int omp_thres_num = blocks.size();
#pragma omp parallel for schedule(dynamic) num_threads(omp_thres_num)
    for (int i = 0; i < blocks.size(); ++ i) {
        auto &block = blocks[i];
        reconstruction->Copy(block->config.Images(), block->mappoints, block->reconstruction);
    }
   
    for (int i = 0; i < blocks.size(); ++ i) {
        auto &block = blocks[i];
        const auto &mappts = block->reconstruction->MapPoints();
        block->mappoints.clear();
        for (const auto &pt : mappts)
        {
            block->mappoints.insert(pt.first);
        }
    }
 
    // fix block images
    if (!options_.use_prior_absolute_location || !reconstruction->b_aligned) {
        for (auto block : blocks) {
            const auto &images = block.second->config.Images();
            auto fst_img_id = *(images.begin());
            block.second->config.RemoveConstantTvec(fst_img_id);
            block.second->config.SetConstantPose(fst_img_id);
            auto snd_img_id = -1;
            for (const auto img : images) {
                if (img == fst_img_id) continue;

                if (correspondence->NumCorrespondencesBetweenImages(img, fst_img_id)) {
                    snd_img_id = img;
                    break;
                }
            }
//            block.second->config.RemoveConstantTvec(snd_img_id);
//            block.second->config.SetConstantPose(snd_img_id);
        }
    }
    ed = std::chrono::steady_clock::now();

    printf("[BATime] split config finished, %d blocks, %.3fs %.3fs %.3fs %.3fs %.3fs\n", 
        blocks.size(), t1, t2, t3, t4, 0.001f * std::chrono::duration<float, std::milli>(ed - st).count());

    for (const auto &block : blocks) {
        const auto &images = block.second->config.Images();
        const auto &com_images = block.second->common_images;
        std::cout << "Block" << block.first << ' ' << images.size() << ' ' << com_images.size() << std::endl;

        /*std::cout << "All images: ";
        for (const auto &img : images) std::cout << img << ' ';
        std::cout << std::endl;

        std::cout << "Common images: ";
        for (const auto &img : com_images) std::cout << img << ' ';
        std::cout << std::endl;*/
    }
    return blocks;
}

size_t BlockBundleAdjuster::Triangulate(Reconstruction* reonstruction, const mappoint_t &mappoint_id) {
    size_t num_tri = 0;
    if (!reonstruction->ExistsMapPoint(mappoint_id)) {
        std::cout << " Do not exist map point ..." << std::endl;
        return num_tri;
    }

    class MapPoint& mappoint = reonstruction->MapPoint(mappoint_id);
    class Track& track = mappoint.Track();

    // Setup data for triangulation estimation.
    std::vector<TriangulationEstimator::PointData> point_data;
    point_data.resize(track.Length());
    std::vector<TriangulationEstimator::PoseData> pose_data;
    pose_data.resize(track.Length());
    for (size_t i = 0; i < track.Length(); ++i) {
        const TrackElement& track_el = track.Elements()[i];
        const Image& image = reonstruction->Image(track_el.image_id);
        const Camera& camera = reonstruction->Camera(image.CameraId());
        const Point2D& point2D = image.Point2D(track_el.point2D_idx);

        point_data[i].point = point2D.XY();
        point_data[i].point_normalized = camera.ImageToWorld(point_data[i].point);
        pose_data[i].proj_matrix = image.ProjectionMatrix();
        pose_data[i].proj_center = image.ProjectionCenter();
        pose_data[i].camera = &camera;

        if(camera.ModelName().compare("SPHERICAL")==0){
            point_data[i].point_bearing = camera.ImageToBearing(point_data[i].point);
        }
    }

    // Setup estimation options.
    EstimateTriangulationOptions tri_options;
    tri_options.min_tri_angle = DegToRad(1.5);
    tri_options.residual_type = TriangulationEstimator::ResidualType::ANGULAR_ERROR;
    tri_options.ransac_options.max_error = DegToRad(2.0);//DegToRad(5.0);
    tri_options.ransac_options.confidence = 0.9999;
    tri_options.ransac_options.min_inlier_ratio = 0.02;
    tri_options.ransac_options.max_num_trials = 10000;

    // Enforce exhaustive sampling for small track lengths.
    const size_t kExhaustiveSamplingThreshold = 15;
    if (point_data.size() <= kExhaustiveSamplingThreshold) {
        tri_options.ransac_options.min_num_trials = NChooseK(point_data.size(), 2);
    }

    // Estimate triangulation.
    Eigen::Vector3d xyz;
    std::vector<char> inlier_mask;
    if (!EstimateTriangulation(tri_options, point_data, pose_data, &inlier_mask, &xyz)) {
        return num_tri;
    }

    mappoint.SetXYZ(xyz);
    num_tri++;

    return num_tri;
}

std::shared_ptr<SimilarityTransform3> BlockBundleAdjuster::EstimateAffine(
    const Reconstruction *reconstruction,
    const std::shared_ptr<Block> src_block, 
    const std::shared_ptr<Block> dst_block)
{
    const double min_inliers = 0.3;
    const double max_reproj_error = 8.0;
    Eigen::Matrix3x4d alignment;
    bool res = ComputeAlignmentBetweenReconstructions(
        *(src_block->reconstruction), 
        *(dst_block->reconstruction), 
        *(reconstruction->GetCorrespondenceGraph()),
        min_inliers, 
        max_reproj_error,
        &alignment);
    
    if (!res) return nullptr;

    return std::make_shared<SimilarityTransform3>(alignment);
}

void BlockBundleAdjuster::BlockPoseGraph(Reconstruction *map_reconstruction, const std::unordered_map<block_t, std::shared_ptr<Block>> &blocks)
{
    auto UpdateImageSim3Pose = [](
        Reconstruction *reconstruction, const image_t img_id) -> bool {
            auto &image = reconstruction->Image(img_id);
            Eigen::Quaterniond quat(image.Qvec()[0], image.Qvec()[1], image.Qvec()[2], image.Qvec()[3]);
            ConvertSIM3tosim3(image.Sim3pose(), quat, image.Tvec(), 1);
            return true;
    };

    std::vector<std::vector<std::shared_ptr<SimilarityTransform3>>> recon_trans(
        blocks.size(),
        std::vector<std::shared_ptr<SimilarityTransform3>>(blocks.size(), nullptr));
    
    /// Get transformation between reconstructions 
    for (const auto src_block : blocks) {
        const auto src = src_block.second;
        for (const auto dst_block : blocks) {
            const auto dst = dst_block.second;
            if (src->id == dst->id) continue;

            const auto inv_trans = recon_trans[dst->id][src->id];
            if (inv_trans != nullptr) {
                recon_trans[src->id][dst->id] = std::make_shared<SimilarityTransform3>(inv_trans->Inverse());
                std::cout << "BATime Affine " << src->id << ' ' << dst->id << ' ' << recon_trans[src->id][dst->id]->Scale() << std::endl;
                continue;
            }

            recon_trans[src->id][dst->id] = EstimateAffine(map_reconstruction, src, dst);
            if (recon_trans[src->id][dst->id] != nullptr) {
		        recon_trans[dst->id][src->id] = std::make_shared<SimilarityTransform3>(recon_trans[src->id][dst->id]->Inverse());

                std::cout << "BATime Affine " << src->id << ' ' << dst->id << ' ' << recon_trans[src->id][dst->id]->Scale() << std::endl;
            } else {
                std::cout << "BATime Affine fail " << src->id << ' ' << dst->id << ' ' << std::endl;
            }
        }
    }

    auto reconstruction = std::make_shared<Reconstruction>();
    for (int i = 0; i < blocks.size(); ++ i) {
        auto img = std::make_shared<Image>();
        img->SetImageId(i);
        
        reconstruction->AddImage(*img);
        UpdateImageSim3Pose(reconstruction.get(), i);
    }

    PoseGraphOptimizer::Options option;
    option.lossfunction_enable = true;
    option.max_num_iterations = 100;
    option.optimization_method = PoseGraphOptimizer::OPTIMIZATION_METHOD::SIM3;
    PoseGraphOptimizer optimizer(option, reconstruction.get());

    for (int i = 0; i < blocks.size(); ++ i) {
        const auto &src_block = blocks.at(i);
        for (int j = i + 1; j < blocks.size(); ++ j) {
            const auto dst_block = blocks.at(j);
            if (recon_trans[src_block->id][dst_block->id] == nullptr) continue;

            auto sim3_src_dst = recon_trans[dst_block->id][src_block->id];
            std::cout << "check " << dst_block->id << ' ' << src_block->id << std::endl;
            std::cout << sim3_src_dst->Matrix() << std::endl;
            double scale = sim3_src_dst->Scale();

            //Eigen::Vector4d diff_r = sim3_src_dst->Rotation();
            //Eigen::Quaterniond diff_q(diff_r[0], diff_r[1], diff_r[2], diff_r[3]);
	        Eigen::Quaterniond diff_q(1.0, 0.0, 0.0, 0.0);
	        Eigen::Vector3d diff_t(0, 0, 0);
            //Eigen::Vector3d diff_t = sim3_src_dst->Translation();
 
            Eigen::Vector7d diff_sim3;
            ConvertSIM3tosim3(diff_sim3, diff_q, diff_t, scale);

            optimizer.AddConstraint(src_block->id, dst_block->id, diff_sim3);
        }
    }

    /*if (blocks.size() >= 3) {
        for (int i = 0; i < blocks.size(); ++ i) {
            const auto &dst_block = blocks.at(i);

            for (int j = i + 1; j < blocks.size(); ++ j) {
                const auto src_block = blocks.at(j);
                if (recon_trans[src_block->id][dst_block->id] == nullptr) continue;

                auto sim3_dst_src = recon_trans[src_block->id][dst_block->id];

                std::string label = "./debug/times_" + std::to_string(ba_times) + "_" + std::to_string(src_block->id) + "_" + std::to_string(dst_block->id) + "_";
                FILE *fp_src = fopen((label + "src_trans.obj").c_str(), "w+");
                const auto &src_mappt = src_block->mappoints;
                for (auto pt : src_mappt) {
                    auto p = src_block->reconstruction->MapPoint(pt).XYZ();

                    sim3_dst_src->TransformPoint(&p);

                    fprintf(fp_src, "v %f %f %f\n", p[0], p[1], p[2]);
                }
                fclose(fp_src);
            }
        }
    }*/

    optimizer.SetParameterConstant(0);
    optimizer.Solve();

    for (int i = 0; i < blocks.size(); ++ i) {
        double scale;
        Eigen::Quaterniond qvec;
        Eigen::Vector3d tvec;
        auto &image = reconstruction->Image(i);
        Convertsim3toSIM3(image.Sim3pose(), qvec, tvec, scale);

	    auto quat = qvec.normalized();
        Eigen::Vector4d n_qvec(quat.w(), quat.x(), quat.y(), quat.z());
        SimilarityTransform3 s_transform(scale, n_qvec, tvec);

	    auto s_transform_inv = s_transform.Inverse();

	    std::cout << "Scale res, block" << i << ' ' << scale << std::endl;
        // apply scale to all block images
        auto block = blocks.at(i);
        const auto &blk_imgs = block->config.Images();
        auto blk_reconstruction = block->reconstruction;
        for (const auto &img : blk_imgs) {
            auto &blk_image = blk_reconstruction->Image(img);
	        s_transform_inv.TransformPose(&blk_image.Qvec(), &blk_image.Tvec());
        }
	
	    const auto &blk_pts = block->mappoints;
        for (const auto &pt : blk_pts) {
            auto xyz = blk_reconstruction->MapPoint(pt).XYZ();

            // Transform the mappoint
            s_transform_inv.TransformPoint(&xyz);
            blk_reconstruction->MapPoint(pt).SetXYZ(xyz);
        }
    }
}

void BlockBundleAdjuster::PoseGraph(
    Reconstruction *reconstruction, 
    std::unordered_map<block_t, std::shared_ptr<Block>> &blocks,
    const BlockBundleAdjustmentOptions &option) 
{
    auto UpdateImagePose = [](
        Reconstruction *reconstruction, const class Image &ref_image) -> bool {
            const image_t img_id = ref_image.ImageId();
            auto &recon_img = reconstruction->Image(img_id);
            recon_img.SetQvec(ref_image.Qvec());
            recon_img.SetTvec(ref_image.Tvec());
            return true;
        };

    auto st = std::chrono::steady_clock::now();
    std::vector<std::vector<std::shared_ptr<SimilarityTransform3>>> recon_trans(
        blocks.size(),
        std::vector<std::shared_ptr<SimilarityTransform3>>(blocks.size(), nullptr));
    
    PoseGraphOptimizer::Options pg_option;
    pg_option.lossfunction_enable = true;
    pg_option.max_num_iterations = 30;
    pg_option.optimization_method = PoseGraphOptimizer::OPTIMIZATION_METHOD::SE3;
    PoseGraphOptimizer optimizer(pg_option, reconstruction);

    std::unordered_set<image_t> visited_images;
    size_t edge_cnt = 0;
    int fst_normal_img = -1;
    
    for (const auto &block : blocks) {
        const auto &blk_imgs = block.second->config.Images();
        const auto &common_imgs = block.second->common_images;
        std::vector<image_t> normal_imgs;
        for (const auto &img : blk_imgs) {
            if (common_imgs.count(img)) continue;
            normal_imgs.emplace_back(img);

            if (fst_normal_img == -1) fst_normal_img = img;
        }
        // add loop edges
        for (const auto &common_img : common_imgs) {
            Eigen::Vector3d t_src, t_dst;
            Eigen::Matrix3d r_src, r_dst;
            const auto &optimized_img = block.second->reconstruction->Image(common_img);
            t_src = optimized_img.Tvec();
            r_src = optimized_img.RotationMatrix();
            if (visited_images.count(common_img) == 0) {
                visited_images.insert(common_img);
                UpdateImagePose(reconstruction, optimized_img);
            }

            std::unordered_set<image_t> pick_normal_imgs;
            while (pick_normal_imgs.size() < option.link_edges) {
                size_t idx = std::rand() % normal_imgs.size();
                pick_normal_imgs.insert(normal_imgs[idx]);
            }
            for (const auto &normal_img : pick_normal_imgs) {
                const auto &optimized_img = block.second->reconstruction->Image(normal_img);
                t_dst = optimized_img.Tvec();
                r_dst = optimized_img.RotationMatrix();

                if (visited_images.count(normal_img) == 0) {
                    visited_images.insert(normal_img);
                    UpdateImagePose(reconstruction, optimized_img);
                }

                Eigen::Matrix3d diff_r = r_src * r_dst.inverse();
                Eigen::Vector3d diff_t = t_src - diff_r * t_dst;

                optimizer.AddConstraint(common_img, normal_img, Eigen::Quaterniond(diff_r), diff_t);

                ++ edge_cnt;
            }
        }

        // add normal edges
        for (const auto &normal_img : normal_imgs) {
            Eigen::Vector3d t_src, t_dst;
            Eigen::Matrix3d r_src, r_dst;
            const auto &optimized_img = block.second->reconstruction->Image(normal_img);
            t_src = optimized_img.Tvec();
            r_src = optimized_img.RotationMatrix();
            if (visited_images.count(normal_img) == 0) {
                visited_images.insert(normal_img);
                UpdateImagePose(reconstruction, optimized_img);
            }

            std::unordered_set<image_t> pick_other_normal_imgs;
            while (pick_other_normal_imgs.size() < option.link_edges) {
                size_t idx = std::rand() % normal_imgs.size();
                if (normal_imgs[idx] == normal_img) continue;

                pick_other_normal_imgs.insert(normal_imgs[idx]);
            }
            for (const auto &another_normal_img : pick_other_normal_imgs) {
                const auto &optimized_img = block.second->reconstruction->Image(another_normal_img);
                t_dst = optimized_img.Tvec();
                r_dst = optimized_img.RotationMatrix();

                if (visited_images.count(another_normal_img) == 0) {
                    visited_images.insert(another_normal_img);
                    UpdateImagePose(reconstruction, optimized_img);
                }

                auto diff_r = r_src * r_dst.inverse();
                auto diff_t = t_src - diff_r * t_dst;
                optimizer.AddConstraint(normal_img, another_normal_img, Eigen::Quaterniond(diff_r), diff_t);
                ++ edge_cnt;
            }
        }
    }

    optimizer.SetParameterConstant(fst_normal_img);

    optimizer.Solve();

    /*std::unordered_map<block_t, EIGEN_STL_UMAP(camera_t, class Camera)> block_cameras;
    for (const auto &block : blocks) {
        block_cameras[block.first] = block.second->reconstruction->Cameras();
    }
    double min_reproj_err = 100000.0;
    block_t best_block_camera = 0;
    for (const auto &block_camera : block_cameras) {
        double reproj_err = 0.0;
        for (auto &block : blocks) {
            for (const auto &camera : block_camera.second) {
                block.second->reconstruction->SetCamera(camera.first, camera.second);
            }
            block.second->reconstruction->FilterAllMapPoints(1, 100000.0, 0.001);
            double block_reproj_err = 0.0;
            const auto &mappoints = block.second->reconstruction->MapPoints();
            for (const auto &pt : mappoints) {
                block_reproj_err += pt.second.Error();
            }
            block_reproj_err /= block.second->config.NumImages();

            reproj_err += block_reproj_err;
            printf("Camera%d, block%d, avg err %lf\n", block_camera.first, block.first, block_reproj_err);
        }

        if (reproj_err < min_reproj_err) {
            min_reproj_err = reproj_err;
            best_block_camera = block_camera.first;
        }
    }
    // update cameras
    printf("BestCamera%d, avg err %lf\n", best_block_camera, min_reproj_err);
    const auto &best_camera = block_cameras[best_block_camera];
    for (const auto &camera : best_camera) {
        reconstruction->SetCamera(camera.first, camera.second);
    }
    for (const auto &camera : block_cameras) {
        auto block = blocks[camera.first];
        const auto &block_camera = camera.second;
        for (const auto &cam: block_camera) {
            block->reconstruction->SetCamera(cam.first, cam.second);
        }
    }*/
    for (const auto &block : blocks) {
        const auto &cameras = block.second->reconstruction->Cameras();
        for (const auto &camera : cameras) {
            reconstruction->SetCamera(camera.first, camera.second);
        }
    }
    // update images : already solved in posegraph
    // update mappoints
    auto sst = std::chrono::steady_clock::now();
    const auto &mappoints = reconstruction->MapPointIds();
    std::vector<mappoint_t> mappoints_vec;
    mappoints_vec.reserve(mappoints.size());
    for (const auto map_id : mappoints) mappoints_vec.emplace_back(map_id);
    size_t system_threads_num = std::thread::hardware_concurrency();
#pragma omp parallel for schedule(dynamic) num_threads(system_threads_num)
    for (int i = 0; i < mappoints_vec.size(); ++ i) {
        mappoint_t map_id = mappoints_vec[i];
        Triangulate(reconstruction, map_id);
    }
    auto eed = std::chrono::steady_clock::now();
    auto trianglate_t = std::chrono::duration<float, std::milli>(eed - sst).count();
    auto ed = std::chrono::steady_clock::now();
    printf("Refine pose, %d nodes, %d edges, %.3f sec.\n",
        visited_images.size(),
        edge_cnt,
        std::chrono::duration<float>(ed - st).count());
}

bool BlockBundleAdjuster::Solve(Reconstruction *reconstruction) {
    if(0) {
        auto blk_recon = std::make_shared<Reconstruction>();
        blk_recon->ReadBinary("/data/xiangxiaojun/AnalyticalDebugData/debugdata/Before_BlockBA/22/2");

        BundleAdjustmentConfig config;
        const auto &images = blk_recon->Images();
        for (const auto &img : images) {
            config.AddImage(img.first);
        }

        image_t fst_img_id, snd_img_id;
        double max_distance = 0;
        for (auto img1 : images) {
            for (auto img2 : images) {
                if (img1.first == img2.first) continue;
                auto position1 = img1.second.ProjectionCenter();
                auto position2 = img2.second.ProjectionCenter();
                auto dist = (position1 - position2).norm();
                if (dist > max_distance) {
                    max_distance = dist;
                    fst_img_id = img1.first;
                    snd_img_id = img2.first;
                }
            }
        }
        config.RemoveConstantTvec(fst_img_id);
        config.SetConstantPose(fst_img_id);
        //config.RemoveConstantTvec(snd_img_id);
        //config.SetConstantPose(snd_img_id);
        options_.parameterize_points_with_track = true;
        BundleAdjuster bundle_adjuster(options_, config);
        bundle_adjuster.Solve(blk_recon.get());
        blk_recon->WriteBinary("./debug/");
        std::cout << "finish" << std::endl;
        exit(0);
    }
    // 1. check
    if (block_options_.block_images_num <= 0 || config_.NumImages() <= block_options_.max_block_images) {
        // Solve
#ifdef USE_OPENBLAS
        openblas_set_num_threads(GetEffectiveNumThreads(-1));
#endif
        BundleAdjuster bundle_adjuster(options_, config_);
        return bundle_adjuster.Solve(reconstruction);
    }
    // never use gps prior in block ba
    options_.use_prior_absolute_location = false;
    options_.plane_constrain = false;
#ifdef USE_OPENBLAS
    openblas_set_num_threads(1);
#endif
    // clear all constant poses
    const auto constant_poses = config_.ConstantPoses();
    for (const auto img : constant_poses) {
        config_.SetVariablePose(img);
    }
    const auto constant_tvecs = config_.ConstantTVecs();
    for (const auto img : constant_tvecs) {
        config_.RemoveConstantTvec(img.first);
    }

    std::string workspace_path = options_.workspace_path;

    if (options_.debug_info) {
        if (global_ba_count == 0) {
            boost::filesystem::remove_all(workspace_path + "/debug/");
            boost::filesystem::remove_all(workspace_path + "/Begin/");
            boost::filesystem::remove_all(workspace_path + "/Before_BlockBA/");
            boost::filesystem::remove_all(workspace_path + "/After_BlockBA/");
            boost::filesystem::remove_all(workspace_path + "/End/");
        }
        std::string rec_path = workspace_path + "/Begin";
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }
        std::string ba_path = rec_path + "/" + std::to_string(global_ba_count);
        if (boost::filesystem::exists(ba_path)) {
            boost::filesystem::remove_all(ba_path);
        }
        boost::filesystem::create_directories(ba_path);
        reconstruction->WriteBinary(ba_path);
    }

    // 2. divide blocks
    auto st = std::chrono::steady_clock::now();
    auto blocks = DivideBlocks(reconstruction, block_options_);
    auto ed = std::chrono::steady_clock::now();
    printf("Divide %d blocks, %.3f sec.\n", blocks.size(), std::chrono::duration<float>(ed - st).count());

    if (blocks.empty()) {
        printf("Divide blocks failed, run full bundle adjustment");
        // Revert configs and solve
        for (const auto img : constant_poses) {
            config_.SetConstantPose(img);
        }
        for (const auto img : constant_tvecs) {
            config_.SetConstantTvec(img.first, img.second);
        }

        /// Err 
        {
            std::string rec_path = StringPrintf("%s/%d_blockba_err",workspace_path.c_str(), reconstruction->NumRegisterImages());
			if (boost::filesystem::exists(rec_path)){
				boost::filesystem::remove_all(rec_path);
			}
			boost::filesystem::create_directories(rec_path);
			reconstruction->WriteBinary(rec_path);
            reconstruction->GetCorrespondenceGraph()->WriteCorrespondenceBinaryData(rec_path + "/corr_graph.bin");
        }
#ifdef USE_OPENBLAS
        openblas_set_num_threads(GetEffectiveNumThreads(-1));
#endif
        BundleAdjuster bundle_adjuster(options_, config_);
        return bundle_adjuster.Solve(reconstruction);
    }
    if (options_.debug_info) {
        std::string debug_dir = workspace_path + "/debug/";
        if (!boost::filesystem::exists(debug_dir)) {
            boost::filesystem::create_directories(debug_dir);
        }
        std::string dir = workspace_path + "/debug/times" + std::to_string(global_ba_count);
        DebugBlocks(dir, blocks);
    }

    if (options_.debug_info) {
        std::string rec_path = workspace_path + "/Before_BlockBA";
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }
        std::string ba_path = rec_path + "/" + std::to_string(global_ba_count);
        if (boost::filesystem::exists(ba_path)) {
            boost::filesystem::remove_all(ba_path);
        }
        boost::filesystem::create_directories(ba_path);
        for (auto block : blocks) {
            std::string blk_path = ba_path + "/" + std::to_string(block.first);
            if (boost::filesystem::exists(blk_path)) {
                boost::filesystem::remove_all(blk_path);
            }
            boost::filesystem::create_directories(blk_path);
            block.second->reconstruction->WriteBinary(blk_path);
       }
    }

    // 3. solve
    st = std::chrono::steady_clock::now();
    options_.parameterize_points_with_track = true;
    int omp_thres_num = CalculateThreadsNum(blocks.size(), block_options_.maximum_threads_num);
    const auto origin_ceres_thread_num = options_.solver_options.num_threads;
    {
        size_t system_threads_num = std::thread::hardware_concurrency();
        if (omp_thres_num * 3 > system_threads_num) options_.solver_options.num_threads = 3;
    }

    // auto BlockBAFunc = [&](std::unordered_map<block_t, std::shared_ptr<Block>>& blocks, block_t i, const BundleAdjustmentOptions& options){
        
    //     auto sst = std::chrono::steady_clock::now();
    //     auto &block = blocks[i];
    //     const auto &blk_config = block->config;
    //     auto blk_reconstruction = block->reconstruction.get();
    //     block_t block_id = block->id;
        
    //     // Solve
    //     BundleAdjuster bundle_adjuster(options, blk_config);
    //     bundle_adjuster.Solve(blk_reconstruction);

    //     blk_reconstruction->FilterAllMapPoints(2, 4.0, 1.5);

    //     const auto &filtered_pts = blk_reconstruction->MapPoints();
    //     block->mappoints.clear();
    //     for (const auto &pt : filtered_pts) {
    //         block->mappoints.insert(pt.first);
    //     }
    //     auto eed = std::chrono::steady_clock::now();
    //     printf("BlockBA block%d %d %d %.3f sec.\n", block_id, block->config.NumImages(), block->mappoints.size(), 
    //     std::chrono::duration<float>(eed - sst).count());
    // };

    // ThreadPool block_ba_thread_pool(omp_thres_num);
    // std::vector<std::future<void>> futures;
    // futures.resize(omp_thres_num);
#pragma omp parallel for schedule(dynamic) num_threads(omp_thres_num)
    for (int i = 0; i < blocks.size(); ++ i) 
    {
        // futures[i] = block_ba_thread_pool.AddTask(BlockBAFunc,blocks,i,options_);
                
        auto sst = std::chrono::steady_clock::now();
        auto &block = blocks[i];
        const auto &blk_config = block->config;
        auto blk_reconstruction = block->reconstruction.get();
        block_t block_id = block->id;
        
        // Solve
        BundleAdjuster bundle_adjuster(options_, blk_config);
        bundle_adjuster.Solve(blk_reconstruction);

        blk_reconstruction->FilterAllMapPoints(2, 4.0, 1.5);

        const auto &filtered_pts = blk_reconstruction->MapPoints();
        block->mappoints.clear();
        for (const auto &pt : filtered_pts) {
            block->mappoints.insert(pt.first);
        }
        auto eed = std::chrono::steady_clock::now();
        printf("BlockBA block%d %d %d %.3f sec.\n", block_id, block->config.NumImages(), block->mappoints.size(), 
        std::chrono::duration<float>(eed - sst).count());
    }
    // for(int i = 0; i<blocks.size(); ++i){
    //     futures[i].get();
    // }
    ed = std::chrono::steady_clock::now();
    printf("BlockBA %d blocks, %.3f sec.\n", blocks.size(), std::chrono::duration<float>(ed - st).count());

    if (options_.debug_info) {
        std::string dir = workspace_path + "/debug/af_blockba_times" + std::to_string(global_ba_count);
        DebugBlocks(dir, blocks);
    }

    if (options_.debug_info) {
        std::string rec_path = workspace_path + "/After_BlockBA";
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }
        std::string ba_path = rec_path + "/" + std::to_string(global_ba_count);
        if (boost::filesystem::exists(ba_path)) {
            boost::filesystem::remove_all(ba_path);
        }
        boost::filesystem::create_directories(ba_path);
        for (auto block : blocks) {
            std::string blk_path = ba_path + "/" + std::to_string(block.first);
            if (boost::filesystem::exists(blk_path)) {
                boost::filesystem::remove_all(blk_path);
            }
            boost::filesystem::create_directories(blk_path);
            block.second->reconstruction->WriteBinary(blk_path);
       }
    }
    // 5. pose graph optimization
    st = std::chrono::steady_clock::now();
    /*BlockPoseGraph(reconstruction, blocks);
    if (block_options_.debug_info) {
        std::string dir = "./debug/af_posegraph_times" + std::to_string(ba_times);
        DebugBlocks(dir, blocks);
    }*/
#ifdef USE_OPENBLAS
    openblas_set_num_threads(GetEffectiveNumThreads(-1));
#endif
    PoseGraph(reconstruction, blocks, block_options_);
    ed = std::chrono::steady_clock::now();
    //printf("Refine pose %.3f sec.\n", blocks.size(), std::chrono::duration<float>(ed - st).count());
#ifdef USE_OPENBLAS
    openblas_set_num_threads(1);
#endif
    // 6. optimized mappoints and cameras, fix images
    st = std::chrono::steady_clock::now();    
    const auto &cameras = reconstruction->Cameras();
    for (const auto &camera: cameras) {
        config_.SetConstantCamera(camera.first);
    }    
    const auto &images = config_.Images();
    for (const auto &img : images) {
        config_.RemoveConstantTvec(img);
        config_.SetConstantPose(img);
    }
    options_.solver_options.num_threads = origin_ceres_thread_num;
    options_.parameterize_points_with_track = false;
    options_.refine_points_only = true;
    BundleAdjuster bundle_adjuster(options_, config_);
    bundle_adjuster.Solve(reconstruction);
    //reconstruction->FilterAllMapPoints(2, 4.0, 1.5);
    ed = std::chrono::steady_clock::now();
    printf("Refine map %.3f sec.\n", blocks.size(), std::chrono::duration<float>(ed - st).count());

    if (options_.debug_info) {
        std::string dir = workspace_path + "/debug/af_mapba_times" + std::to_string(global_ba_count);
        DebugReconstruction(dir, reconstruction);
    }

    if (options_.debug_info) {
        std::string rec_path = workspace_path + "/End";
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }
        std::string ba_path = rec_path + "/" + std::to_string(global_ba_count);
        if (boost::filesystem::exists(ba_path)) {
            boost::filesystem::remove_all(ba_path);
        }
        boost::filesystem::create_directories(ba_path);
        reconstruction->WriteBinary(ba_path);
    }
    return true;
}

size_t BlockBundleAdjuster::CalculateThreadsNum(const size_t blocks_count, const size_t threads_thres) const
{
    size_t system_threads_num = std::thread::hardware_concurrency();

    if (system_threads_num <= 0) return 1;

    if (threads_thres <= 0) {
        return std::min(blocks_count, system_threads_num);
    }

    return std::min(std::min(threads_thres, blocks_count), system_threads_num);
}

}  // namespace sensemap
