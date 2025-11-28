//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_FEATURE_MATCHER_H_
#define SENSEMAP_FEATURE_MATCHER_H_

#include <array>
#include <string>
#include <vector>

#include "util/threading.h"
#include "feature/matching_utils.h"
#include "container/feature_data_container.h"
#include "container/match_data_container.h"
#include "container/scene_graph_container.h"

namespace sensemap{

enum MatchMethod {
    NONE = 0,
    EXHAUSTIVE,
    SEQUENTIAL,
    VOCABTREE,
    SPATIAL,
    HYBRID,
    HYBRID_INPUT,
    MANUAL
};

struct ExhaustiveMatchingOptions {
    // Block size, i.e. number of images to simultaneously load into memory.
    int block_size = 50;
    bool Check() const;
};

struct SequentialMatchingOptions {

    // Number of overlapping image pairs.
    int overlap = 10;

    // Whether to match images against their quadratic neighbors.
    bool quadratic_overlap = false;

    //loop detection with multiple checks
    bool robust_loop_detection = false;

    //loop detection before sequential matching
    bool loop_detection_before_sequential_matching = false;

    bool local_triplet_checking = false;

    // Whether to enable vocabulary tree based loop detection.
    bool loop_detection = false;

    // Loop detection is invoked every `loop_detection_period` images.
    int loop_detection_period = 10;

    // The number of images to retrieve in loop detection. This number should
    // be significantly bigger than the sequential matching overlap.
    int loop_detection_num_images = 50;

    // Only if the score of a neighbor is larger than a certain ratio of the 
    // best score, it would be kept as a loop candidate. 
    double loop_detection_min_score_ratio_to_best = 0.5;

    // Number of nearest neighbors to retrieve per query feature.
    int loop_detection_num_nearest_neighbors = 1;

    // Number of nearest-neighbor checks to use in retrieval.
    int loop_detection_num_checks = 256;

    // How many images to return after spatial verification. Set to 0 to turn 
    // off spatial verification.
    int loop_detection_num_images_after_verification = 0;

    // The maximum number of features to use for indexing an image. If an
    // image has more features, only the largest-scale features will be indexed.
    int loop_detection_max_num_features = -1;

    // Number of threads for loop detection
    int loop_detection_num_threads = 1;

    float local_max_recent_score_factor = 0.8f;

    float local_best_acc_score_factor = 0.75f;

    // int local_loop_consistency_threshold = 3;

    int local_region_repetitive = 0;

    float max_recent_score_factor = 0.8f;

    float best_acc_score_factor = 0.75f;

    int loop_consistency_threshold = 3;

    // Path to the vocabulary tree.
    std::string vocab_tree_path = "";

    // Path to the vlad codebook
    std::string vlad_code_book_path = "";

    bool Check() const;
};

struct VocabTreeMatchingOptions {

    //Number of threads to index and retrieval
    int num_threads = -1;

    // Number of images to retrieve for each query image.
    int num_images = 50;

    // Only if the score of a neighbor is larger than a certain ratio of the 
    // best score, it would be kept as a match candidate. 
    double min_score_ratio_to_best = 0.5;

    // Number of nearest neighbors to retrieve per query feature.
    int num_nearest_neighbors = 15;

    // Number of nearest-neighbor checks to use in retrieval.
    int num_checks = 256;

    // How many images to return after spatial verification. Set to 0 to turn
    // off spatial verification.
    int num_images_after_verification = 0;

    float max_score_factor = 0.0f;

    // The maximum number of features to use for indexing an image. If an
    // image has more features, only the largest-scale features will be indexed.
    int vocab_tree_max_num_features = -1;

    // Path to the vocabulary tree.
    std::string vocab_tree_path = "";

     // Path to the vlad codebook
    std::string vlad_code_book_path = "";
    
    // Optional path to file with specific image names to match.
    std::string match_list_path = "";

    bool Check() const;
};

struct SpatialMatchingOptions {
    //Number of threads
    int num_threads = -1;
    // Whether the location priors in the database are GPS coordinates in
    // the form of longitude and latitude coordinates in degrees.
    bool is_gps = true;

    // Whether to ignore the Z-component of the location prior.
    bool ignore_z = false;

    // The maximum number of nearest neighbors to match.
    int max_num_neighbors = 20;

    // The maximum distance between the query and nearest neighbor. For GPS
    // coordinates the unit is Euclidean distance in meters.
    double max_distance = 50;

    bool Check() const;
};

struct HybridMatchingOptions {
    // For Hybird Matching, we would perform cluster partition by image label
    // Define the match method inside each cluster
    // -- Set default inside cluster matching method as sequential
    MatchMethod method_inside_cluster = MatchMethod::SEQUENTIAL; 

    // Define the match method outside each cluster
    // -- Set default outside cluster matching method as vocabulary tree
    MatchMethod method_outside_cluster = MatchMethod::VOCABTREE; 

    bool Check() const;
};

struct ManualMatchingOptions {
    std::function<std::vector<image_t>(image_t, const FeatureDataContainer &)> generate_pairs_for_image;

    bool Check() const;
};

// the overall feature matching options
struct FeatureMatchingOptions {

public:
    enum RetrieveType {
        SIFT = 0,
        VLAD = 1
    };
    typedef sensemap::MatchMethod MatchMethod;

    ExhaustiveMatchingOptions ExhaustiveMatching() const;
    SequentialMatchingOptions SequentialMatching () const;
    VocabTreeMatchingOptions VocabTreeMatching() const;
    SpatialMatchingOptions SpatialMatching() const;
    HybridMatchingOptions HybridMatching() const;
    ManualMatchingOptions ManualMatching() const;
    PairMatchingOptions PairMatching() const;
    MatchMethod Method() const;
    bool Check() const;

public:
    
    bool have_prior_image_pairs_ = false;
    std::vector<std::pair<image_t,image_t>> prior_image_pairs_;

    bool have_prior_pose_ = false;
    double prior_neighbor_distance = std::numeric_limits<double>::max();
    std::unordered_map<image_t, Eigen::Vector4d> prior_rotations;
    std::unordered_map<image_t, Eigen::Vector3d> prior_translations;
    std::unordered_map<image_t, bool> prior_pose_validations;

    bool have_prior_bluetooth_ = false;
    
    std::unordered_map<double, std::vector<std::pair<std::string, int>>> prior_bluetooth_time_signal;
    double prior_bluetooth_threshold_inside = 3;
    double prior_bluetooth_threshold_outside = 7;
    double prior_bluetooth_threshold_outlier = 9;

    bool have_prior_location_ = false;
    double max_distance_for_loop = std::numeric_limits<double>::max();
    std::unordered_map<image_t, std::pair<Eigen::Vector3d,int>> prior_locations;
    
    int min_track_degree = 3;
    int max_cover_per_view = 800;
    int select_range_match_point = 0;
    bool track_preoperation = true;

    int track_block_radius = 200;
    int track_max_per_block = 30;
    int track_min_per_block = 20;
    int track_max_cover_per_view = 5000;

    // The max distance for image pairs that can be matched, this is a ratio to
    // prior_neighbor_distance
    double max_match_distance = 80.0;

    Eigen::Matrix3x4d ned_to_ecef_matrix_;

    bool global_triplet_checking = false;

    float local_invalid_theta_dis = 10.0f;
    float global_median_invalid_theta_dis = 10.0f;
    float global_mean_invalid_theta_dis = 8.0f;

    int ambiguous_triple_count = 2;

    bool detect_apriltag_ = false;

    bool match_between_reconstructions_ = false;

    bool delete_duplicated_images_ = true;

    // remove ambiguous feature match in map update mode.
	bool map_update = false;

    ExhaustiveMatchingOptions exhaustive_matching_;
    SequentialMatchingOptions sequential_matching_;
    VocabTreeMatchingOptions vocabtree_matching_;
    SpatialMatchingOptions spatial_matching_;
    HybridMatchingOptions hybrid_matching_;
    PairMatchingOptions pair_matching_;
    ManualMatchingOptions manual_matching_;
    MatchMethod method_=EXHAUSTIVE;
    RetrieveType retrieve_type = SIFT;
};

//This base class defines a single thread for matching of image pairs
class PairsMatcherThread : public Thread {
public:
    PairsMatcherThread(const PairMatchingOptions& options,
                       FeatureDataContainer* container);

    void SetMaxNumMatches(const int max_num_matches);

protected:
    PairMatchingOptions options_;
    FeatureDataContainer* feature_data_container_;
};

//Normal matching of image pairs on CPU
class CPUPairsMatcherThread : public PairsMatcherThread {
public:
    typedef FeatureMatcherData Input;
    typedef FeatureMatcherData Output;

CPUPairsMatcherThread(const PairMatchingOptions& options,
                      FeatureDataContainer* container,
                      JobQueue<Input>* input_queue,
                      JobQueue<Output>* output_queue);

protected:
    void Run() override;

    JobQueue<Input>* input_queue_;
    JobQueue<Output>* output_queue_;
};

//Normal matching of image pairs on GPU
class GPUPairsMatcherThread : public PairsMatcherThread {
public:
    typedef FeatureMatcherData Input;
    typedef FeatureMatcherData Output;

GPUPairsMatcherThread(const PairMatchingOptions& options,
                      FeatureDataContainer* container,
                      JobQueue<Input>* input_queue,
                      JobQueue<Output>* output_queue);

    static int obj_count;
protected:
    void Run() override;
    void GetDescriptorData(const int index, const image_t image_id,
                           const CompressedFeatureDescriptors** descriptors_ptr);

    JobQueue<Input>* input_queue_;
    JobQueue<Output>* output_queue_;


    std::array<image_t, 2> prev_uploaded_image_ids_;
    std::array<CompressedFeatureDescriptors, 2> prev_uploaded_descriptors_;
};

//Guided matching of image pairs on CPU
class GuidedCPUPairsMatcherThread : public PairsMatcherThread {
public:
    typedef FeatureMatcherData Input;
    typedef FeatureMatcherData Output;

    GuidedCPUPairsMatcherThread(const PairMatchingOptions& options,
                                FeatureDataContainer* container,
                                JobQueue<Input>* input_queue,
                                JobQueue<Output>* output_queue);

private:
    void Run() override;

    JobQueue<Input>* input_queue_;
    JobQueue<Output>* output_queue_;
};

//guided matching of image pairs on GPU
class GuidedGPUPairsMatcherThread : public PairsMatcherThread {
public:
    typedef FeatureMatcherData Input;
    typedef FeatureMatcherData Output;

    GuidedGPUPairsMatcherThread(const PairMatchingOptions& options,
                                FeatureDataContainer* container,
                                JobQueue<Input>* input_queue,
                                JobQueue<Output>* output_queue);

private:
    void Run() override;
    void GetFeatureData(const int index, const image_t image_id,
                        const FeatureKeypoints** keypoints_ptr,
                        const CompressedFeatureDescriptors** descriptors_ptr);

    JobQueue<Input>* input_queue_;
    JobQueue<Output>* output_queue_;


    std::array<image_t, 2> prev_uploaded_image_ids_;
    std::array<FeatureKeypoints, 2> prev_uploaded_keypoints_;
    std::array<CompressedFeatureDescriptors, 2> prev_uploaded_descriptors_;
};


//This class defines a thread to verify the matching results, and computes the
//geometry between a pair of images. 
class TwoViewGeometryVerifier : public Thread {
public:
    typedef FeatureMatcherData Input;
    typedef FeatureMatcherData Output;

    TwoViewGeometryVerifier(const PairMatchingOptions& options,
                            FeatureDataContainer* container,
                            JobQueue<Input>* input_queue,
                            JobQueue<Output>* output_queue);

protected:
    void Run() override;

    const PairMatchingOptions options_;
    TwoViewGeometry::Options two_view_geometry_options_;
    FeatureDataContainer* feature_data_container_;
    JobQueue<Input>* input_queue_;
    JobQueue<Output>* output_queue_;
};

//Multi-thread image pairs matcher, matching the image pairs generated by
//PairsGenerator. These pairs are collected into
//an input queue, and the results are received by an output queue. Several
//intemediate processes, i.e. plain matching, geometric verifying, guided 
//matching, enft tracking and re-verifying after enft, are concatenated
//to accomplish this task. 

class PairsMatcher {
public:
    PairsMatcher(const PairMatchingOptions& options,
                 FeatureDataContainer* container);

    ~PairsMatcher();

    // Setup the matchers and return if successful.
    bool Setup();

    // Match a batch of multiple image pairs.
    void Match(const std::vector<std::pair<image_t, image_t>>& image_pairs,
               MatchDataContainer* match_data_container);


private:
    PairMatchingOptions options_;
    FeatureDataContainer* feature_data_container_;

    bool is_setup_;

    std::vector<std::unique_ptr<PairsMatcherThread>> match_threads_;
    std::vector<std::unique_ptr<PairsMatcherThread>> guided_match_threads_;
    std::vector<std::unique_ptr<Thread>> verifiers_;

    JobQueue<FeatureMatcherData> matcher_queue_;
    JobQueue<FeatureMatcherData> verifier_queue_;
    JobQueue<FeatureMatcherData> guided_matcher_queue_;
    JobQueue<FeatureMatcherData> output_queue_;
};

//This class generates image pairs to be matched. According to
//the options, pairs are generated exhaustively, sequentially, or by
//similarities in either appearances or locations.

class PairsGenerator{
public:
    PairsGenerator(const FeatureMatchingOptions& options,
                   FeatureDataContainer* container);

    void Run();
    const std::vector<std::pair<image_t, image_t>>& GetPairs();
    const std::vector<std::pair<image_t, image_t>>& GetLoopPairs();
private:

    void GeneratePairsExhaustively(
        const std::vector<image_t> &input_image_id = std::vector<image_t>());
    void GeneratePairsSequentially(
        const std::vector<image_t> &input_image_id = std::vector<image_t>());
    void GeneratePairsVocabTree(
        const std::vector<image_t> &input_image_id = std::vector<image_t>());

    void GeneratePairsRobustLoopDetection(
        const std::vector<image_t> &input_image_id = std::vector<image_t>());
    void GeneratePairsHybridly();
    void GeneratePairsWithHybridInput();

    void GeneratePairsManually();

    void GeneratePairsSpatially(
            const std::vector<image_t> &input_image_id = std::vector<image_t>());

    // Generate Match pair by April Tags Detection
    void GeneratePairsByAprilTag();
    // Generate pair between new images and images from old reconstruction
    void GeneratePairsBetweenReconVocabTree();

    FeatureMatchingOptions options_;    
    FeatureDataContainer* feature_data_container_;
    std::vector<std::pair<image_t,image_t>> image_pairs_;
    std::vector<std::pair<image_t,image_t>> loop_image_pairs_;
    int hybrid_status_;

private:
    enum HYBRID_STATUS{
        // -- Hybrid matching is disabled
        DISABLE, 
        // -- Hybrid matching is enable with matching inside each cluster
        INSIDE_MATCHING,
        // -- Hybrid matching is enable with matching between clusters   
        OUTSIDE_MATCHING 
    };
};

//The matching process are decomposed into two parts, first generating a set of
//image pairs and then matching these pairs. 

class FeatureMatcher{
public:
    FeatureMatcher(const FeatureMatchingOptions& options,
                   FeatureDataContainer* container,
                   MatchDataContainer* md_container,
                   SceneGraphContainer* graph);
    
    void Run();
    void BuildSceneGraph();
    std::unordered_set<image_pair_t> GetStrongLoops(){return strong_loop_pairs_;}

private:
    int BluetoothBeaconDistance(const double& timestamp1, const double& timestamp2, double& distance);
    void CheckLoopMatches(const std::vector<std::pair<image_t,image_t>>& loop_image_pairs_, bool check = false);
    void SequentialLoopClosing();

    FeatureMatchingOptions options_;
    PairsGenerator generator_;
    PairsMatcher matcher_;
    FeatureDataContainer* feature_data_container_;
    SceneGraphContainer* scene_graph_;
    MatchDataContainer* match_data_container_;
    std::vector<bool> inlier_matches_;
    std::unordered_set<image_pair_t> strong_loop_pairs_;
};

}//namespace sensemap

#endif//SENSEMAP_FEATURE_MATCHER_H