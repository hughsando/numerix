#pragma once

#define COS__0PI_OVER_32 1.0f

#ifdef HX_WINDOWS

#define SQRT2_OVER_2 0.707106781186548f
#define SQRT2_OVER_4 0.353553390593274f

#define COS__1PI_OVER_32 0.995184726672197f
#define COS__2PI_OVER_32 0.98078528040323f
#define COS__3PI_OVER_32 0.956940335732209f
#define COS__4PI_OVER_32 0.923879532511287f
#define COS__5PI_OVER_32 0.881921264348355f
#define COS__6PI_OVER_32 0.831469612302545f
#define COS__7PI_OVER_32 0.773010453362737f
#define COS__8PI_OVER_32 0.707106781186548f
#define COS__9PI_OVER_32 0.634393284163645f
#define COS_10PI_OVER_32 0.555570233019602f
#define COS_11PI_OVER_32 0.471396736825998f
#define COS_12PI_OVER_32 0.38268343236509f
#define COS_13PI_OVER_32 0.290284677254462f
#define COS_14PI_OVER_32 0.195090322016128f
#define COS_15PI_OVER_32 0.0980171403295608f

#else

#define SQRT2_OVER_2 0x1.6A09E6p-1f
#define SQRT2_OVER_4 0x1.6A09E6p-2f

#define COS__1PI_OVER_32 0x1.FD88DAp-1f
#define COS__2PI_OVER_32 0x1.F6297Cp-1f
#define COS__3PI_OVER_32 0x1.E9F416p-1f
#define COS__4PI_OVER_32 0x1.D906BCp-1f
#define COS__5PI_OVER_32 0x1.C38B30p-1f
#define COS__6PI_OVER_32 0x1.A9B662p-1f
#define COS__7PI_OVER_32 0x1.8BC806p-1f
#define COS__8PI_OVER_32 SQRT2_OVER_2
#define COS__9PI_OVER_32 0x1.44CF32p-1f
#define COS_10PI_OVER_32 0x1.1C73B4p-1f
#define COS_11PI_OVER_32 0x1.E2B5D4p-2f
#define COS_12PI_OVER_32 0x1.87DE2Ap-2f
#define COS_13PI_OVER_32 0x1.294062p-2f
#define COS_14PI_OVER_32 0x1.8F8B84p-3f
#define COS_15PI_OVER_32 0x1.917A6Cp-4f

#endif


#define COS_16PI_OVER_32 0.0f
#define COS_17PI_OVER_32 -COS_15PI_OVER_32
#define COS_18PI_OVER_32 -COS_14PI_OVER_32
#define COS_19PI_OVER_32 -COS_13PI_OVER_32
#define COS_20PI_OVER_32 -COS_12PI_OVER_32
#define COS_21PI_OVER_32 -COS_11PI_OVER_32
#define COS_22PI_OVER_32 -COS_10PI_OVER_32
#define COS_23PI_OVER_32 -COS__9PI_OVER_32
#define COS_24PI_OVER_32 -SQRT2_OVER_2
#define COS_25PI_OVER_32 -COS__7PI_OVER_32
#define COS_26PI_OVER_32 -COS__6PI_OVER_32
#define COS_27PI_OVER_32 -COS__5PI_OVER_32
#define COS_28PI_OVER_32 -COS__4PI_OVER_32
#define COS_29PI_OVER_32 -COS__3PI_OVER_32
#define COS_30PI_OVER_32 -COS__2PI_OVER_32
#define COS_31PI_OVER_32 -COS__1PI_OVER_32

#define COS__0PI_OVER_16 1.0f
#define COS__1PI_OVER_16 COS__2PI_OVER_32
#define COS__2PI_OVER_16 COS__4PI_OVER_32
#define COS__3PI_OVER_16 COS__6PI_OVER_32
#define COS__4PI_OVER_16 SQRT2_OVER_2
#define COS__5PI_OVER_16 COS_10PI_OVER_32
#define COS__6PI_OVER_16 COS_12PI_OVER_32
#define COS__7PI_OVER_16 COS_14PI_OVER_32
#define COS__8PI_OVER_16 0.0f
#define COS__9PI_OVER_16 -COS__7PI_OVER_16 
#define COS_10PI_OVER_16 -COS__6PI_OVER_16
#define COS_11PI_OVER_16 -COS__5PI_OVER_16
#define COS_12PI_OVER_16 -SQRT2_OVER_2
#define COS_13PI_OVER_16 -COS__3PI_OVER_16
#define COS_14PI_OVER_16 -COS__2PI_OVER_16
#define COS_15PI_OVER_16 -COS__1PI_OVER_16

#define COS_0PI_OVER_8 1.0f
#define COS_1PI_OVER_8 COS__4PI_OVER_32
#define COS_2PI_OVER_8 SQRT2_OVER_2
#define COS_3PI_OVER_8 COS_12PI_OVER_32
#define COS_4PI_OVER_8 0.0f
#define COS_5PI_OVER_8 -COS_3PI_OVER_8
#define COS_6PI_OVER_8 -SQRT2_OVER_2
#define COS_7PI_OVER_8 -COS_1PI_OVER_8

#define COS_0PI_OVER_4 1.0f
#define COS_1PI_OVER_4 SQRT2_OVER_2
#define COS_2PI_OVER_4 0.0f
#define COS_3PI_OVER_4 -SQRT2_OVER_2

#define COS_0PI_OVER_2 1.0f
#define COS_1PI_OVER_2 0.0f

#define SIN__0PI_OVER_32 0.0f
#define SIN__1PI_OVER_32 COS_15PI_OVER_32
#define SIN__2PI_OVER_32 COS_14PI_OVER_32
#define SIN__3PI_OVER_32 COS_13PI_OVER_32
#define SIN__4PI_OVER_32 COS_12PI_OVER_32
#define SIN__5PI_OVER_32 COS_11PI_OVER_32
#define SIN__6PI_OVER_32 COS_10PI_OVER_32
#define SIN__7PI_OVER_32 COS__9PI_OVER_32
#define SIN__8PI_OVER_32 SQRT2_OVER_2
#define SIN__9PI_OVER_32 COS__7PI_OVER_32
#define SIN_10PI_OVER_32 COS__6PI_OVER_32
#define SIN_11PI_OVER_32 COS__5PI_OVER_32
#define SIN_12PI_OVER_32 COS__4PI_OVER_32
#define SIN_13PI_OVER_32 COS__3PI_OVER_32
#define SIN_14PI_OVER_32 COS__2PI_OVER_32
#define SIN_15PI_OVER_32 COS__1PI_OVER_32
#define SIN_16PI_OVER_32 1.0f
#define SIN_17PI_OVER_32 COS__1PI_OVER_32
#define SIN_18PI_OVER_32 COS__2PI_OVER_32
#define SIN_19PI_OVER_32 COS__3PI_OVER_32
#define SIN_20PI_OVER_32 COS__4PI_OVER_32
#define SIN_21PI_OVER_32 COS__5PI_OVER_32
#define SIN_22PI_OVER_32 COS__6PI_OVER_32
#define SIN_23PI_OVER_32 COS__7PI_OVER_32
#define SIN_24PI_OVER_32 SQRT2_OVER_2
#define SIN_25PI_OVER_32 COS__9PI_OVER_32
#define SIN_26PI_OVER_32 COS_10PI_OVER_32
#define SIN_27PI_OVER_32 COS_11PI_OVER_32
#define SIN_28PI_OVER_32 COS_12PI_OVER_32
#define SIN_29PI_OVER_32 COS_13PI_OVER_32
#define SIN_30PI_OVER_32 COS_14PI_OVER_32
#define SIN_31PI_OVER_32 COS_15PI_OVER_32

#define SIN__0PI_OVER_16 0.0f
#define SIN__1PI_OVER_16 COS__7PI_OVER_16
#define SIN__2PI_OVER_16 COS__6PI_OVER_16
#define SIN__3PI_OVER_16 COS__5PI_OVER_16
#define SIN__4PI_OVER_16 SQRT2_OVER_2
#define SIN__5PI_OVER_16 COS__3PI_OVER_16
#define SIN__6PI_OVER_16 COS__2PI_OVER_16
#define SIN__7PI_OVER_16 COS__1PI_OVER_16
#define SIN__8PI_OVER_16 1.0f
#define SIN__9PI_OVER_16 COS__1PI_OVER_16
#define SIN_10PI_OVER_16 COS__2PI_OVER_16
#define SIN_11PI_OVER_16 COS__3PI_OVER_16
#define SIN_12PI_OVER_16 SQRT2_OVER_2
#define SIN_13PI_OVER_16 COS__5PI_OVER_16
#define SIN_14PI_OVER_16 COS__6PI_OVER_16
#define SIN_15PI_OVER_16 COS__7PI_OVER_16

#define SIN_0PI_OVER_8 0.0f
#define SIN_1PI_OVER_8 COS_3PI_OVER_8
#define SIN_2PI_OVER_8 SQRT2_OVER_2
#define SIN_3PI_OVER_8 COS_1PI_OVER_8
#define SIN_4PI_OVER_8 1.0f
#define SIN_5PI_OVER_8 COS_1PI_OVER_8
#define SIN_6PI_OVER_8 SQRT2_OVER_2
#define SIN_7PI_OVER_8 COS_3PI_OVER_8

#define SIN_0PI_OVER_4 0.0f
#define SIN_1PI_OVER_4 SQRT2_OVER_2
#define SIN_2PI_OVER_4 1.0f
#define SIN_3PI_OVER_4 SQRT2_OVER_2

#define SIN_0PI_OVER_2 0.0f
#define SIN_1PI_OVER_2 1.0f

#define COMPLEX_LITERAL_HELPER(real, imag) (real + imag##i)
#define COMPLEX_LITERAL(real, imag) COMPLEX_LITERAL_HELPER(real, imag)

#define CEXP__0PI_OVER_16 COMPLEX_LITERAL(COS__0PI_OVER_16, SIN__0PI_OVER_16)
#define CEXP__1PI_OVER_16 COMPLEX_LITERAL(COS__1PI_OVER_16, SIN__1PI_OVER_16)
#define CEXP__2PI_OVER_16 COMPLEX_LITERAL(COS__2PI_OVER_16, SIN__2PI_OVER_16)
#define CEXP__3PI_OVER_16 COMPLEX_LITERAL(COS__3PI_OVER_16, SIN__3PI_OVER_16)
#define CEXP__4PI_OVER_16 COMPLEX_LITERAL(COS__4PI_OVER_16, SIN__4PI_OVER_16)
#define CEXP__5PI_OVER_16 COMPLEX_LITERAL(COS__5PI_OVER_16, SIN__5PI_OVER_16)
#define CEXP__6PI_OVER_16 COMPLEX_LITERAL(COS__6PI_OVER_16, SIN__6PI_OVER_16)
#define CEXP__7PI_OVER_16 COMPLEX_LITERAL(COS__7PI_OVER_16, SIN__7PI_OVER_16)
#define CEXP__8PI_OVER_16 COMPLEX_LITERAL(COS__8PI_OVER_16, SIN__8PI_OVER_16)
#define CEXP__9PI_OVER_16 COMPLEX_LITERAL(COS__9PI_OVER_16, SIN__9PI_OVER_16)
#define CEXP_10PI_OVER_16 COMPLEX_LITERAL(COS_10PI_OVER_16, SIN_10PI_OVER_16)
#define CEXP_11PI_OVER_16 COMPLEX_LITERAL(COS_11PI_OVER_16, SIN_11PI_OVER_16)
#define CEXP_12PI_OVER_16 COMPLEX_LITERAL(COS_12PI_OVER_16, SIN_12PI_OVER_16)
#define CEXP_13PI_OVER_16 COMPLEX_LITERAL(COS_13PI_OVER_16, SIN_13PI_OVER_16)
#define CEXP_14PI_OVER_16 COMPLEX_LITERAL(COS_14PI_OVER_16, SIN_14PI_OVER_16)
#define CEXP_15PI_OVER_16 COMPLEX_LITERAL(COS_15PI_OVER_16, SIN_15PI_OVER_16)

#define CEXP_0PI_OVER_8 COMPLEX_LITERAL(COS_0PI_OVER_8, SIN_0PI_OVER_8)
#define CEXP_1PI_OVER_8 COMPLEX_LITERAL(COS_1PI_OVER_8, SIN_1PI_OVER_8)
#define CEXP_2PI_OVER_8 COMPLEX_LITERAL(COS_2PI_OVER_8, SIN_2PI_OVER_8)
#define CEXP_3PI_OVER_8 COMPLEX_LITERAL(COS_3PI_OVER_8, SIN_3PI_OVER_8)
#define CEXP_4PI_OVER_8 COMPLEX_LITERAL(COS_4PI_OVER_8, SIN_4PI_OVER_8)
#define CEXP_5PI_OVER_8 COMPLEX_LITERAL(COS_5PI_OVER_8, SIN_5PI_OVER_8)
#define CEXP_6PI_OVER_8 COMPLEX_LITERAL(COS_6PI_OVER_8, SIN_6PI_OVER_8)
#define CEXP_7PI_OVER_8 COMPLEX_LITERAL(COS_7PI_OVER_8, SIN_7PI_OVER_8)

#define CEXP_0PI_OVER_4 COMPLEX_LITERAL(COS_0PI_OVER_4, SIN_0PI_OVER_4)
#define CEXP_1PI_OVER_4 COMPLEX_LITERAL(COS_1PI_OVER_4, SIN_1PI_OVER_4)
#define CEXP_2PI_OVER_4 COMPLEX_LITERAL(COS_2PI_OVER_4, SIN_2PI_OVER_4)
#define CEXP_3PI_OVER_4 COMPLEX_LITERAL(COS_3PI_OVER_4, SIN_3PI_OVER_4)

#define CEXP_0PI_OVER_2 COMPLEX_LITERAL(COS_0PI_OVER_2, SIN_0PI_OVER_2)
#define CEXP_1PI_OVER_2 COMPLEX_LITERAL(COS_1PI_OVER_2, SIN_1PI_OVER_2)
