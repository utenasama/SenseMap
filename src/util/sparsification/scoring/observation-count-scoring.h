#ifndef MAP_SPARSIFICATION_HEURISTIC_SCORING_OBSERVATION_COUNT_SCORING_H_
#define MAP_SPARSIFICATION_HEURISTIC_SCORING_OBSERVATION_COUNT_SCORING_H_

#include "scoring-function.h"

namespace sensemap {
namespace scoring {

class ObservationCountScoringFunction : public ScoringFunction {
public:
    virtual ~ObservationCountScoringFunction() {}

private:
    virtual inline double scoreImpl(
            const mappoint_t store_landmark_id,
            const Reconstruction &map) const {
        CHECK(map.ExistsMapPoint(store_landmark_id));
        return map.MapPoint(store_landmark_id).Track().Length();
    }
};

}  // namespace scoring
}  // namespace sensemap
#endif  // MAP_SPARSIFICATION_HEURISTIC_SCORING_OBSERVATION_COUNT_SCORING_H_
