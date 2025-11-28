//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "util/bitmap.h"
#include "util/graph.h"

namespace sensemap {

namespace graph {
void ExportToGraph(const std::string& filename,
                   const Eigen::MatrixXi& data) {
    int min_val = data.minCoeff();
    int max_val = data.maxCoeff();

    Bitmap bitmap;
    // int max_size = 1600;
    // if(data.cols() * 4 > max_size){
    //     max_size = data.cols() * 4;
    // }
    // int scale = max_size / data.cols();
    // bitmap.Allocate(data.cols() * scale, data.rows() * scale, true);
    bitmap.Allocate(data.cols(), data.rows(), true);
    bitmap.Fill(BitmapColor<uint8_t>(255));
    const double max_value = std::log1p(max_val);
    for (int i = 0; i < data.rows(); ++i) {
        for (int j = 0; j < data.cols(); ++j) {
            const double value = std::log1p(data(i, j)) / max_value;
            const BitmapColor<float> color(255 * JetColormap::Red(value),
                                           255 * JetColormap::Green(value),
                                           255 * JetColormap::Blue(value));
            bitmap.SetPixel(j, i, color.Cast<uint8_t>());
            // for(int ii = 0; ii < scale; ++ii) {
            //     for(int ij = 0; ij <scale; ++ij) {
            //         bitmap.SetPixel(j * scale + ij , i * scale + ii,
            //                         color.Cast<uint8_t>());
            //     }
            // }

//            uint8_t val = 255 * (data(i, j) - min_val) / (max_val - min_val);
//            if (data(i, j) > 0) {
//                val = val < 100 ? 100 : val;
//            }
//            BitmapColor<uint8_t> color(val, val, val);
//            bitmap.SetPixel(j, i, val);
        }
    }
    bitmap.Write(filename);
}
} // namespace graph

} // namespace sensemap