//Copyright (c) 2022, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_UTIL_COLOR_SPACE_H_
#define SENSEMAP_UTIL_COLOR_SPACE_H_

#define INV_COLOR_NORM 0.003921569

namespace sensemap {

struct YCrCbFactor {
    double s_Y, s_Cb, s_Cr;
    double o_Y, o_Cb, o_Cr;
};

template <typename T>
void
ColorRGBToYCbCr(T* v) {
    T out[3];
    out[0] = v[0] * T(0.299) + v[1] * T(0.587) + v[2] * T(0.114);
    out[1] = v[0] * T(-0.168736) + v[1] * T(-0.331264) + v[2] * T(0.5) + T(0.5);
    out[2] = v[0] * T(0.5) + v[1] * T(-0.418688) + v[2] * T(-0.081312) + T(0.5);
    std::copy_n(out, 3, v);
}

template <typename T>
void
ColorYCbCrToRGB (T* v) {
    v[1] = v[1] - T(0.5);
    v[2] = v[2] - T(0.5);

    T out[3];
    out[0] = v[0] * T(1) + v[1] * T(0) + v[2] * T(1.402);
    out[1] = v[0] * T(1) + v[1] * T(-0.34414) + v[2] * T(-0.71414);
    out[2] = v[0] * T(1) + v[1] * T(1.772) + v[2] * T(0);
    std::copy_n(out, 3, v);
}
}

#endif