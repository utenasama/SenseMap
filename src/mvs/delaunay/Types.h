////////////////////////////////////////////////////////////////////
// Types.h
//
// Copyright 2007 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#ifndef SENSEMAP_MVS_DELAUNAY_TYPES_H_
#define SENSEMAP_MVS_DELAUNAY_TYPES_H_

#define RESTRICT          __restrict__

#define ZERO_TOLERANCE    (1e-7)
#define INV_ZERO        (1e+14)

#define FZERO_TOLERANCE    0.0001f
#define FINV_ZERO        1000000.f

#define GCLASS            unsigned
#define FRONT            0
#define BACK            1
#define PLANAR_DEL        2
#define CLIPPED            3
#define CULLED            4
#define VISIBLE            5

#ifdef _DEBUG

#ifdef _MSC_VER
#define _DEBUGINFO
#define _CRTDBG_MAP_ALLOC    //enable this to show also the filename (DEBUG_NEW should also be defined in each file)
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _INC_CRTDBG
#define ASSERT(exp)    {if (!(exp) && 1 == _CrtDbgReport(_CRT_ASSERT, __FILE__, __LINE__, NULL, #exp)) _CrtDbgBreak();}
#else
#define ASSERT(exp)    {if (!(exp)) __debugbreak();}
#endif // _INC_CRTDBG
#define TRACE(...) {TCHAR buffer[2048];    _sntprintf(buffer, 2048, __VA_ARGS__); OutputDebugString(buffer);}
#else // _MSC_VER
#include <assert.h>
#define ASSERT(exp)    assert(exp)
#define TRACE(...)
#endif // _MSC_VER

#else

#ifdef _RELEASE
#define ASSERT(exp)
#else
#ifdef _MSC_VER
#define ASSERT(exp) {if (!(exp)) __debugbreak();}
#else // _MSC_VER
#define ASSERT(exp) {if (!(exp)) __builtin_trap();}
#endif // _MSC_VER
#endif
#define TRACE(...)

#endif // _DEBUG

#ifndef STCALL
# if defined(_MSC_VER)
#  define STCALL __cdecl
# elif defined(__OS2__)
#  if defined (__GNUC__) && __GNUC__ < 4
#   define STCALL _cdecl
#  else
#   /* On other compilers on OS/2, we use the _System calling convention */
#   /* to be compatible with every compiler */
#   define STCALL _System
#  endif
# elif defined(__GNUC__)
#   define STCALL __attribute__((__cdecl__))
# else
#  define STCALL
# endif
#endif // STCALL

#define ASSERTM(exp, msg) ASSERT(exp)

#include <algorithm>

#ifndef MINF
#define MINF            std::min
#endif
#ifndef MAXF
#define MAXF            std::max
#endif

template<typename _Tp>
inline _Tp    ZEROTOLERANCE()                { return _Tp(0); }

#define DECLARE_NO_INDEX(...) std::numeric_limits<__VA_ARGS__>::max()
#define NO_ID                DECLARE_NO_INDEX(uint32_t)

#include <stdint.h>
#include <cmath>
#include <limits>

typedef int32_t                HRESULT;

typedef unsigned char        BYTE;
typedef unsigned short        WORD;
typedef unsigned int        DWORD;
typedef uint64_t            QWORD;

typedef char                CHAR;
typedef CHAR*                LPSTR;
typedef const CHAR*            LPCSTR;
typedef CHAR                TCHAR;
typedef LPSTR                LPTSTR;
typedef LPCSTR                LPCTSTR;

typedef float REAL;

template<typename _Tp>
inline _Tp    ABS(_Tp    x)                    { return std::abs(x); }

template<typename T>
constexpr T& NEGATE(T& a) {
    return (a = -a);
}
template<typename T>
constexpr T SQUARE(const T& a) {
    return (a * a);
}
template<typename T>
constexpr T CUBE(const T& a) {
    return (a * a * a);
}
template<typename T>
inline T SQRT(const T& a) {
    return T(std::sqrt(a));
}
template<typename T>
inline T EXP(const T& a) {
    return T(std::exp(a));
}
template<typename T>
inline T LOGN(const T& a) {
    return T(std::log(a));
}
template<typename T>
inline T LOG10(const T& a) {
    return T(std::log10(a));
}

inline bool   ISINFORNAN(float x)            { return (std::isinf(x) || std::isnan(x)); }
inline bool   ISINFORNAN(double x)            { return (std::isinf(x) || std::isnan(x)); }
inline bool   ISFINITE(float x)                { return (!std::isinf(x) && !std::isnan(x)); }
inline bool   ISFINITE(double x)            { return (!std::isinf(x) && !std::isnan(x)); }

template<typename _Tp>
inline bool   ISFINITE(_Tp x)            { return (!std::isinf((double)x) && !std::isnan((double)x)); }

template<typename _Tp>
inline bool   ISFINITE(const _Tp* x, size_t n)    { for (size_t i=0; i<n; ++i) if (ISINFORNAN(x[i])) return false; return true; }

template<typename _Tp>
inline _Tp    CLAMP(_Tp v, _Tp c0, _Tp c1)    { ASSERT(c0<=c1); return MINF(MAXF(v, c0), c1); }

// ISEQUAL
inline bool   ISEQUAL(float  x, float  v)    { return ABS(x-v) < FZERO_TOLERANCE; }
inline bool   ISEQUAL(double x, double v)    { return ABS(x-v) < ZERO_TOLERANCE; }

inline float  INVZERO(float)                { return FINV_ZERO; }
inline double INVZERO(double)                { return INV_ZERO; }
template<typename _Tp>
inline _Tp    INVZERO(_Tp)                    { return std::numeric_limits<_Tp>::max(); }

template<typename _Tp>
inline _Tp    SAFEDIVIDE(_Tp   x, _Tp   y)    { return (y==_Tp(0) ? INVZERO(y) : x/y); }

#endif