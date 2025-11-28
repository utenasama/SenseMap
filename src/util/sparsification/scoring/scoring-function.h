#ifndef MAP_SPARSIFICATION_HEURISTIC_SCORING_SCORING_FUNCTION_H_
#define MAP_SPARSIFICATION_HEURISTIC_SCORING_SCORING_FUNCTION_H_

#include "base/reconstruction.h"
#include "util/types.h"

namespace sensemap {
namespace scoring {

class ScoringFunction {
public:
    SENSEMAP_POINTER_TYPEDEFS(ScoringFunction);

    ScoringFunction() : weight_(1.0) {}

    virtual ~ScoringFunction() {}

    double operator()(
            const mappoint_t store_landmark_id,
            const Reconstruction &map) const {
        double raw_score = scoreImpl(store_landmark_id, map);
        return weight_ * raw_score;
    }

    void setWeight(double weight) {
        weight_ = weight;
    }

private:
    virtual double scoreImpl(
            mappoint_t store_landmark_id,
            const Reconstruction &map) const = 0;

    double weight_;
};

}  // namespace scoring
}  // namespace sensemap
#endif  // MAP_SPARSIFICATION_HEURISTIC_SCORING_SCORING_FUNCTION_H_
