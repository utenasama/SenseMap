#include "sampler-factory.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>

#include "cost-functions/min-keypoints-per-keyframe-cost.h"
#include "heuristic-sampling.h"
#include "scoring/observation-count-scoring.h"
#include "descriptor-scoring-functions/descriptor-variance-scoring.h"

namespace sensemap {

DEFINE_int32(sparsification_min_keypoints_per_keyframe, 30,
             "Minimum desired number of keypoints per each map keyframe");
DEFINE_double(sparsification_descriptor_dev_scoring_threshold, 10,
              "Minimum desired number of keypoints per each map keyframe");

SamplerBase::Ptr createSampler(SamplerBase::Type sampler_type) {
    typedef sampling::LandmarkSamplingWithCostFunctions HeuristicSampler;

    SamplerBase::Ptr sampler;
    switch (sampler_type) {
        case SamplerBase::Type::kHeuristic: {
            HeuristicSampler::Ptr heuristic_sampler(new HeuristicSampler);

            using cost_functions::IsRequiredToConstrainKeyframesCost;
            using descriptorscoring::DescriptorVarianceScoring;
            using scoring::ObservationCountScoringFunction;

            DescriptorVarianceScoring::Ptr descriptor_dev_score(
                    new DescriptorVarianceScoring(FLAGS_sparsification_descriptor_dev_scoring_threshold));
            IsRequiredToConstrainKeyframesCost::Ptr keyframe_keypoint_cost(
                    new IsRequiredToConstrainKeyframesCost(FLAGS_sparsification_min_keypoints_per_keyframe));
            ObservationCountScoringFunction::Ptr obs_count_score(new ObservationCountScoringFunction);

            const double kDescriptorVarianceWeight = 0.2;
            const double kObsCountWeight = 1.0;
            const double kKeyframeConstraintWeight = 1.0;
            descriptor_dev_score->setWeight(kDescriptorVarianceWeight);
            obs_count_score->setWeight(kObsCountWeight);
            keyframe_keypoint_cost->setWeight(kKeyframeConstraintWeight);

            // Squared loss.
            std::function<double(double)> keyframe_constraint_loss =  // NOLINT
                    [](double x) {
                        const double kLinearFactor = 5.0;
                        return kLinearFactor * x * x;
                    };  // NOLINT
            keyframe_keypoint_cost->setLossFunction(keyframe_constraint_loss);

            heuristic_sampler->registerScoringFunction(obs_count_score);
            heuristic_sampler->registerDescriptorScoringFunction(descriptor_dev_score);
            heuristic_sampler->registerCostFunction(keyframe_keypoint_cost);

            sampler = heuristic_sampler;
        }
            break;
            //            case SamplerBase::Type::kLpsolveIlp: {
            //                sampler.reset(
            //                        new LpSolveSparsification(
            //                                FLAGS_sparsification_min_keypoints_per_keyframe));
            //            }
            //                break;
            //            case SamplerBase::Type::kLpsolvePartitionIlp: {
            //                SamplerBase::Ptr ilp_sampler =
            //                        createSampler(SamplerBase::Type::kLpsolveIlp);
            //                CHECK(ilp_sampler != nullptr);
            //                sampler.reset(new GraphPartitionSampler(ilp_sampler));
            //            }
            //                break;
        default:
            std::cout << "Unknown landmark sampler type: "
                      << static_cast<std::underlying_type<SamplerBase::Type>::type>(sampler_type);
    }
    return sampler;
}

}  // namespace sensemap
