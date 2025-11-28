//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_BASE_VISIBILITY_PYRAMID_H_
#define SENSEMAP_BASE_VISIBILITY_PYRAMID_H_

#include <vector>

#include "util/types.h"

namespace sensemap {

class VisibilityPyramid {
    public:
        VisibilityPyramid();
        VisibilityPyramid(const size_t num_levels, const size_t width, const size_t height);

        void SetPoint(const double x, const double y);
        void ResetPoint(const double x, const double y);

        inline size_t NumLevels() const;
        inline size_t Width() const;
        inline size_t Height() const;

        inline size_t Score() const;
        inline size_t MaxScore() const;

    private:
        void CellForPoint(const double x, const double y, size_t* cx, size_t* cy) const;

    private:
        // Range of the input points.
        size_t width_;
        size_t height_;

        // The overall visibility score.
        size_t score_;

        // The maximum score when all cells are populated.
        size_t max_score_;

        // The visibilty pyramid with multiple levels.
        std::vector<Eigen::MatrixXi> pyramid_;

};

//////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
//////////////////////////////////////////////////////////////////////////////////////////////////

size_t VisibilityPyramid::NumLevels() const { return pyramid_.size(); }

size_t VisibilityPyramid::Width() const { return width_; }

size_t VisibilityPyramid::Height() const { return height_; }

size_t VisibilityPyramid::Score() const { return score_; }

size_t VisibilityPyramid::MaxScore() const { return max_score_; }

}

#endif
