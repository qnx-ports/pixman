
/*
 * Copyright © 2024 Filip Wasil, Samsung Electronics
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Author: Filip Wasil (f.wasil@samsung.com)
 */

#ifndef __PIXMAN_RVV_BLENDINGS_H__
#define __PIXMAN_RVV_BLENDINGS_H__

#include <riscv_vector.h>
#include <stddef.h>
#include "pixman-private.h"
#include "pixman-combine-float.h"

/*
 * Screen
 *
 *      ad * as * B(d/ad, s/as)
 *    = ad * as * (d/ad + s/as - s/as * d/ad)
 *    = ad * s + as * d - s * d
 */


__attribute__((always_inline))
inline static
vfloat32m1_t rvv_blend_screen(const vfloat32m1_t sa,
                        const vfloat32m1_t s,
                        const vfloat32m1_t da,
                        const vfloat32m1_t d,
                        size_t vl) {
    register vfloat32m1_t t0,t1,t2;
    t0 = __riscv_vfmul_vv_f32m1(s,da,vl);
    t1 = __riscv_vfmul_vv_f32m1(d,sa,vl);
    t2 = __riscv_vfmul_vv_f32m1(s,d,vl);
    return __riscv_vfsub_vv_f32m1(__riscv_vfadd_vv_f32m1(t0,t1,vl), t2, vl);
}

/*
 * Multiply
 *
 *      ad * as * B(d / ad, s / as)
 *    = ad * as * d/ad * s/as
 *    = d * s
 *
 */


__attribute__((always_inline))
inline static vfloat32m1_t
rvv_blend_multiply(const vfloat32m1_t sa,
                        const vfloat32m1_t s,
                        const vfloat32m1_t da,
                        const vfloat32m1_t d,
                        size_t vl)
{
    return __riscv_vfmul_vv_f32m1(s,d,vl);
}


/*
 * Overlay
 *
 *     ad * as * B(d/ad, s/as)
 *   = ad * as * Hardlight (s, d)
 *   = if (d / ad < 0.5)
 *         as * ad * Multiply (s/as, 2 * d/ad)
 *     else
 *         as * ad * Screen (s/as, 2 * d / ad - 1)
 *   = if (d < 0.5 * ad)
 *         as * ad * s/as * 2 * d /ad
 *     else
 *         as * ad * (s/as + 2 * d / ad - 1 - s / as * (2 * d / ad - 1))
 *   = if (2 * d < ad)
 *         2 * s * d
 *     else
 *         ad * s + 2 * as * d - as * ad - ad * s * (2 * d / ad - 1)
 *   = if (2 * d < ad)
 *         2 * s * d
 *     else
 *         as * ad - 2 * (ad - d) * (as - s)
 */


__attribute__((always_inline))
inline static vfloat32m1_t
rvv_blend_overlay(const vfloat32m1_t sa,
                        const vfloat32m1_t s,
                        const vfloat32m1_t da,
                        const vfloat32m1_t d,
                        size_t vl)
{
    register vfloat32m1_t t0,t1,t2,t3,t4,f0,f1,f2;
    vbool32_t vb;
    t0 = __riscv_vfadd_vv_f32m1(d,d,vl);
    t1 = __riscv_vfmul_vv_f32m1(__riscv_vfadd_vv_f32m1(s,s,vl),d,vl);
    vb = __riscv_vmflt_vv_f32m1_b32(t0,da,vl);
    t2 = __riscv_vfmul_vv_f32m1(sa,da,vl);
    f2 = __riscv_vfsub_vv_f32m1(da,d,vl);
    t3 = __riscv_vfmul_vf_f32m1(f2,2.0f,vl);
    t4 = __riscv_vfsub_vv_f32m1(sa,s,vl);
    f0 = __riscv_vfmul_vv_f32m1(t3,t4,vl);
    f1 = __riscv_vfsub_vv_f32m1(t2,f0,vl);
    return __riscv_vmerge_vvm_f32m1(f1,t1,vb,vl);
}

/*
 * Darken
 *
 *     ad * as * B(d/ad, s/as)
 *   = ad * as * MIN(d/ad, s/as)
 *   = MIN (as * d, ad * s)
 */


__attribute__((always_inline))
inline static vfloat32m1_t
rvv_blend_darken(const vfloat32m1_t sa,
                        const vfloat32m1_t s,
                        const vfloat32m1_t da,
                        const vfloat32m1_t d,
                        size_t vl)
{
    register vfloat32m1_t ss,dd;
    vbool32_t vb;
    ss = __riscv_vfmul_vv_f32m1(da,s,vl);
    dd = __riscv_vfmul_vv_f32m1(sa,d,vl);
    vb = __riscv_vmfgt_vv_f32m1_b32(ss,dd,vl);
    return __riscv_vmerge_vvm_f32m1(ss,dd,vb,vl);
}

/*
 * Lighten
 *
 *     ad * as * B(d/ad, s/as)
 *   = ad * as * MAX(d/ad, s/as)
 *   = MAX (as * d, ad * s)
 */


__attribute__((always_inline))
inline static vfloat32m1_t
rvv_blend_lighten(const vfloat32m1_t sa,
                        const vfloat32m1_t s,
                        const vfloat32m1_t da,
                        const vfloat32m1_t d,
                        size_t vl)
{
    register vfloat32m1_t ss,dd;
    ss  = __riscv_vfmul_vv_f32m1(s,da,vl);
    dd  = __riscv_vfmul_vv_f32m1(d,sa,vl);
    vbool32_t vb = __riscv_vmfgt_vv_f32m1_b32(ss,dd,vl);
    return __riscv_vmerge_vvm_f32m1(dd, ss,vb,vl);
}

/*
 * Color dodge
 *
 *     ad * as * B(d/ad, s/as)
 *   = if d/ad = 0
 *         ad * as * 0
 *     else if (d/ad >= (1 - s/as)
 *         ad * as * 1
 *     else
 *         ad * as * ((d/ad) / (1 - s/as))
 *   = if d = 0
 *         0
 *     elif as * d >= ad * (as - s)
 *         ad * as
 *     else
 *         as * (as * d / (as - s))
 *
 */


__attribute__((always_inline))
inline static vfloat32m1_t
rvv_blend_color_dodge(const vfloat32m1_t sa,
                        const vfloat32m1_t s,
                        const vfloat32m1_t da,
                        const vfloat32m1_t d,
                        size_t vl)
{
    register vfloat32m1_t t0,t1,t2,t3,t4, zero;
    vbool32_t is_d_non_zero, vb, is_t0_non_zero;

    zero = __riscv_vfmv_s_f_f32m1(0.0f, vl);
    is_d_non_zero = __riscv_vmfne_vf_f32m1_b32(d, 0.0f, vl);

    t0 = __riscv_vfsub_vv_f32m1(sa, s, vl); // sa - s
    t1 = __riscv_vfmul_vv_f32m1(sa, d, vl); // d * sa
    t2 = __riscv_vfmul_vv_f32m1(sa, da, vl); // sa * da
    t3 = __riscv_vfsub_vv_f32m1(t2, __riscv_vfmul_vv_f32m1(s, da, vl), vl); // sa * da - s * da

    is_t0_non_zero = __riscv_vmfne_vf_f32m1_b32(t0, 0.0f, vl);
    vb = __riscv_vmflt_vv_f32m1_b32(t3, t1, vl);
    t4 = __riscv_vfdiv_vv_f32m1(__riscv_vfmul_vv_f32m1(sa, t1, vl), t0, vl); // sa * sa * d / (sa - s);

    return __riscv_vmerge_vvm_f32m1(zero,
                __riscv_vmerge_vvm_f32m1(__riscv_vmerge_vvm_f32m1(t2, t4, is_t0_non_zero,vl),t2,
     vb, vl), is_d_non_zero, vl);
}

/*
 * Color burn
 *
 * We modify the first clause "if d = 1" to "if d >= 1" since with
 * premultiplied colors d > 1 can actually happen.
 *
 *     ad * as * B(d/ad, s/as)
 *   = if d/ad >= 1
 *         ad * as * 1
 *     elif (1 - d/ad) >= s/as
 *         ad * as * 0
 *     else
 *         ad * as * (1 - ((1 - d/ad) / (s/as)))
 *   = if d >= ad
 *         ad * as
 *     elif as * ad - as * d >= ad * s
 *         0
 *     else
 *         ad * as  - as * as * (ad - d) / s
 */


__attribute__((always_inline))
inline static vfloat32m1_t
rvv_blend_color_burn(const vfloat32m1_t sa,
                        const vfloat32m1_t s,
                        const vfloat32m1_t da,
                        const vfloat32m1_t d,
                        size_t vl)
{
    register vfloat32m1_t t0,t1,t2,t3,t4,zero;
    vbool32_t is_d_lt_da, is_s_not_zero, vb;

    zero = __riscv_vfmv_s_f_f32m1(0.0f, 1);

    is_d_lt_da = __riscv_vmflt_vv_f32m1_b32(d, da, vl);
    is_s_not_zero = __riscv_vmfne_vf_f32m1_b32(s, 0.0f, vl);

    t0 = __riscv_vfmul_vv_f32m1(sa,__riscv_vfsub_vv_f32m1(da,d,vl),vl); // sa * (da - d)
    t1 = __riscv_vfsub_vv_f32m1(da, __riscv_vfdiv_vv_f32m1(t0, s, vl), vl); // da - sa * (da - d) / s)
    t2 = __riscv_vfmul_vv_f32m1(sa, da, vl); // sa * da
    t3 = __riscv_vfmul_vv_f32m1(sa, t1, vl); // sa * (da - sa * (da - d) / s)
    t4 = __riscv_vfmul_vv_f32m1(s, da, vl); // s * da
    vb = __riscv_vmflt_vv_f32m1_b32(t0, t4, vl); // if (sa * (da - d) < s * da)

    return __riscv_vmerge_vvm_f32m1(t2,
                __riscv_vmerge_vvm_f32m1(
                        zero, __riscv_vmerge_vvm_f32m1(zero, t3, is_s_not_zero,vl), vb, vl),is_d_lt_da,vl);
}

/*
 * Hard light
 *
 *     ad * as * B(d/ad, s/as)
 *   = if (s/as <= 0.5)
 *         ad * as * Multiply (d/ad, 2 * s/as)
 *     else
 *         ad * as * Screen (d/ad, 2 * s/as - 1)
 *   = if 2 * s <= as
 *         ad * as * d/ad * 2 * s / as
 *     else
 *         ad * as * (d/ad + (2 * s/as - 1) + d/ad * (2 * s/as - 1))
 *   = if 2 * s <= as
 *         2 * s * d
 *     else
 *         as * ad - 2 * (ad - d) * (as - s)
 */


__attribute__((always_inline))
inline static vfloat32m1_t
rvv_blend_hard_light(const vfloat32m1_t sa,
                        const vfloat32m1_t s,
                        const vfloat32m1_t da,
                        const vfloat32m1_t d,
                        size_t vl)
{
    register vfloat32m1_t t0,t1,t2,t3,t4;
    vbool32_t vb;
    t0 = __riscv_vfadd_vv_f32m1(s,s,vl);
    t1 = __riscv_vfmul_vv_f32m1(__riscv_vfadd_vv_f32m1(s,s,vl),d,vl);
    vb = __riscv_vmfgt_vv_f32m1_b32(t0,sa,vl);
    t2 = __riscv_vfmul_vv_f32m1(sa,da,vl);
    t3 = __riscv_vfmul_vf_f32m1(__riscv_vfsub_vv_f32m1(da,d,vl),2.0f,vl);
    t4 = __riscv_vfsub_vv_f32m1(sa,s,vl);
    return __riscv_vmerge_vvm_f32m1(t1, __riscv_vfsub_vv_f32m1(t2,__riscv_vfmul_vv_f32m1(t3,t4,vl),vl),vb,vl);
}

/*
 * Soft light
 *
 *     ad * as * B(d/ad, s/as)
 *   = if (s/as <= 0.5)
 *         ad * as * (d/ad - (1 - 2 * s/as) * d/ad * (1 - d/ad))
 *     else if (d/ad <= 0.25)
 *         ad * as * (d/ad + (2 * s/as - 1) * ((((16 * d/ad - 12) * d/ad + 4) * d/ad) - d/ad))
 *     else
 *         ad * as * (d/ad + (2 * s/as - 1) * sqrt (d/ad))
 *   = if (2 * s <= as)
 *         d * as - d * (ad - d) * (as - 2 * s) / ad;
 *     else if (4 * d <= ad)
 *         (2 * s - as) * d * ((16 * d / ad - 12) * d / ad + 3);
 *     else
 *         d * as + (sqrt (d * ad) - d) * (2 * s - as);
 */


__attribute__((always_inline))
inline static vfloat32m1_t
rvv_blend_soft_light(const vfloat32m1_t sa,
                        const vfloat32m1_t s,
                        const vfloat32m1_t da,
                        const vfloat32m1_t d,
                        size_t vl)
{
    register vfloat32m1_t t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13;
    vbool32_t            is_sa_lt_2s, is_da_ls_4d, is_da_non_zero;
    is_da_non_zero = __riscv_vmfne_vf_f32m1_b32(da, 0.0f, vl);
    t0 = __riscv_vfadd_vv_f32m1(s,s,vl); // 2 * s
    is_sa_lt_2s = __riscv_vmflt_vv_f32m1_b32(sa, t0, vl);
    t1 = __riscv_vfmul_vv_f32m1(sa,d,vl); // d * sa
    t2 = __riscv_vfsub_vv_f32m1(sa, t0,vl); // (sa - 2*s)
    t3 = __riscv_vfmul_vv_f32m1(d,t2,vl); // (sa - 2*s) * d
    t7 = __riscv_vfdiv_vv_f32m1(__riscv_vfmul_vf_f32m1(d, 16.0f, vl), da, vl); // 16 * d / da
    t8 = __riscv_vfmul_vv_f32m1(d, __riscv_vfsub_vf_f32m1(t7, 12.0f, vl), vl); // (16 * d / da - 12) * d
    t9 = __riscv_vfadd_vf_f32m1(__riscv_vfdiv_vv_f32m1(t8,da,vl), 3.0f, vl); // (16 * d / da - 12) * d / da + 3)
    t4 = __riscv_vfmul_vv_f32m1(t3,t9,vl); // (sa - 2*s) * d * ((16 * d / da - 12) * d / da + 3)
    t5 = __riscv_vfsub_vv_f32m1(t1,t4,vl); // d * sa - (sa - 2*s) * d * ((16 * d / da - 12) * d / da + 3)
    t6 = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(d,d,vl),__riscv_vfadd_vv_f32m1(d,d,vl), vl);
    is_da_ls_4d = __riscv_vmflt_vv_f32m1_b32(da, t6, vl);
    t10 = __riscv_vfsub_vv_f32m1(__riscv_vfsqrt_v_f32m1(__riscv_vfmul_vv_f32m1(d, da, vl), vl), d, vl); // sqrtf (d * da) - d
    t11 = __riscv_vfmul_vv_f32m1(t2, t10, vl); // (sqrtf (d * da) - d) * (sa - 2 * s)
    t12 = __riscv_vfsub_vv_f32m1(t1, t11, vl); // d * sa - (sqrtf (d * da) - d) * (sa - 2 * s)
    // d * sa - d * (da - d) * (sa - 2 * s) / da
    t13 = __riscv_vfsub_vv_f32m1(t1,
             __riscv_vfdiv_vv_f32m1(
                     __riscv_vfmul_vv_f32m1(
                             __riscv_vfmul_vv_f32m1(d,t2, vl), __riscv_vfsub_vv_f32m1(da, d, vl),vl)
                     ,da,vl),vl);
    return __riscv_vmerge_vvm_f32m1(t1, // if (!FLOAT_IS_ZERO (da))
        __riscv_vmerge_vvm_f32m1(t13, // if (4 * d > da)
        __riscv_vmerge_vvm_f32m1(t5, t12, is_da_ls_4d ,vl), is_sa_lt_2s, vl),is_da_non_zero, vl);
}

/*
 * Difference
 *
 *     ad * as * B(s/as, d/ad)
 *   = ad * as * abs (s/as - d/ad)
 *   = if (s/as <= d/ad)
 *         ad * as * (d/ad - s/as)
 *     else
 *         ad * as * (s/as - d/ad)
 *   = if (ad * s <= as * d)
 *        as * d - ad * s
 *     else
 *        ad * s - as * d
 */


__attribute__((always_inline))
inline static vfloat32m1_t
rvv_blend_difference(const vfloat32m1_t sa,
                        const vfloat32m1_t s,
                        const vfloat32m1_t da,
                        const vfloat32m1_t d,
                        size_t vl)
{
    register vfloat32m1_t dsa,sda;
    vbool32_t            vb;
    dsa  = __riscv_vfmul_vv_f32m1(d,sa,vl);
    sda  = __riscv_vfmul_vv_f32m1(s,da,vl);
    vb = __riscv_vmflt_vv_f32m1_b32(sda,dsa,vl);
    return __riscv_vmerge_vvm_f32m1(__riscv_vfsub_vv_f32m1(sda,dsa,vl),
        __riscv_vfsub_vv_f32m1(dsa,sda,vl),vb,vl);
}

/*
 * Exclusion
 *
 *     ad * as * B(s/as, d/ad)
 *   = ad * as * (d/ad + s/as - 2 * d/ad * s/as)
 *   = as * d + ad * s - 2 * s * d
 */


__attribute__((always_inline))
inline static vfloat32m1_t
rvv_blend_exclusion(const vfloat32m1_t sa,
                        const vfloat32m1_t s,
                        const vfloat32m1_t da,
                        const vfloat32m1_t d,
                        size_t vl)
{
    register vfloat32m1_t t0,t1;
    t0 = __riscv_vfmul_vv_f32m1(__riscv_vfadd_vv_f32m1(d,d,vl),s,vl);
    t1 = __riscv_vfadd_vv_f32m1(__riscv_vfmul_vv_f32m1(s,da,vl), __riscv_vfmul_vv_f32m1(d,sa,vl),vl);
    return __riscv_vfsub_vv_f32m1(t1,t0,vl);
}

typedef vfloat32m1_t (* rvv_combine_channel_float_t) (
                                       const vfloat32m1_t sa,
                                       const vfloat32m1_t s,
                                       const vfloat32m1_t da,
                                       const vfloat32m1_t d,
                                       size_t vl);


__attribute__((always_inline))
inline static void
rvv_combine_inner(pixman_bool_t component,
           float *dest, const float * src, const float *mask, int n_pixels,
           rvv_combine_channel_float_t combine_a, rvv_combine_channel_float_t combine_c)
{
    float * __restrict__       pd = dest;
    const float * __restrict__ ps = src;
    const float * __restrict__ pm = mask;

    const int component_count = 4;
    int vn = component_count*n_pixels;
    int vl = 0;
    int vl_step = 0;
    //const float stride = 4 * sizeof(float);
    const ptrdiff_t stride = 4*sizeof(float);
    if (!mask)
    {
        for (; vn > 0; vn -= vl_step, pd += vl_step, ps += vl_step)
        {
        vl = __riscv_vsetvl_e32m1(vn);

        vfloat32m1_t sa = __riscv_vlse32_v_f32m1(&ps[0],stride,vl);
        vfloat32m1_t sr = __riscv_vlse32_v_f32m1(&ps[1],stride,vl);
        vfloat32m1_t sg = __riscv_vlse32_v_f32m1(&ps[2],stride,vl);
        vfloat32m1_t sb = __riscv_vlse32_v_f32m1(&ps[3],stride,vl);

        vfloat32m1_t da = __riscv_vlse32_v_f32m1(&pd[0],stride,vl);
        vfloat32m1_t dr = __riscv_vlse32_v_f32m1(&pd[1],stride,vl);
        vfloat32m1_t dg = __riscv_vlse32_v_f32m1(&pd[2],stride,vl);
        vfloat32m1_t db = __riscv_vlse32_v_f32m1(&pd[3],stride,vl);

        vfloat32m1_t da2 = combine_a (sa, sa, da, da, vl);
        vfloat32m1_t dr2 = combine_c (sa, sr, da, dr, vl);
        vfloat32m1_t dg2 = combine_c (sa, sg, da, dg, vl);
        vfloat32m1_t db2 = combine_c (sa, sb, da, db, vl);

        __riscv_vsse32_v_f32m1(&pd[0],stride,da2,vl);
        __riscv_vsse32_v_f32m1(&pd[1],stride,dr2,vl);
        __riscv_vsse32_v_f32m1(&pd[2],stride,dg2,vl);
        __riscv_vsse32_v_f32m1(&pd[3],stride,db2,vl);

        vl_step = vl * component_count;
        }
    }
    else
    {
	if (component) {
            for (; vn > 0; vn -= vl_step, pd += vl_step, ps += vl_step, pm += vl_step)
            {
                vl = __riscv_vsetvl_e32m1(vn);

                vfloat32m1_t sa = __riscv_vlse32_v_f32m1(&ps[0],stride,vl);
                vfloat32m1_t sr = __riscv_vlse32_v_f32m1(&ps[1],stride,vl);
                vfloat32m1_t sg = __riscv_vlse32_v_f32m1(&ps[2],stride,vl);
                vfloat32m1_t sb = __riscv_vlse32_v_f32m1(&ps[3],stride,vl);

                vfloat32m1_t ma = __riscv_vlse32_v_f32m1(&pm[0],stride,vl);
                vfloat32m1_t mr = __riscv_vlse32_v_f32m1(&pm[1],stride,vl);
                vfloat32m1_t mg = __riscv_vlse32_v_f32m1(&pm[2],stride,vl);
                vfloat32m1_t mb = __riscv_vlse32_v_f32m1(&pm[3],stride,vl);

                vfloat32m1_t da = __riscv_vlse32_v_f32m1(&pd[0],stride,vl);
                vfloat32m1_t dr = __riscv_vlse32_v_f32m1(&pd[1],stride,vl);
                vfloat32m1_t dg = __riscv_vlse32_v_f32m1(&pd[2],stride,vl);
                vfloat32m1_t db = __riscv_vlse32_v_f32m1(&pd[3],stride,vl);

                vfloat32m1_t sr2 = __riscv_vfmul_vv_f32m1(sr, mr, vl);
                vfloat32m1_t sg2 = __riscv_vfmul_vv_f32m1(sg, mg, vl);
                vfloat32m1_t sb2 = __riscv_vfmul_vv_f32m1(sb, mb, vl);

                vfloat32m1_t ma2 = __riscv_vfmul_vv_f32m1(ma, sa, vl);
                vfloat32m1_t mr2 = __riscv_vfmul_vv_f32m1(mr, sa, vl);
                vfloat32m1_t mg2 = __riscv_vfmul_vv_f32m1(mg, sa, vl);
                vfloat32m1_t mb2 = __riscv_vfmul_vv_f32m1(mb, sa, vl);

                vfloat32m1_t da2 = combine_a (ma2, ma2, da, da, vl);
                vfloat32m1_t dr2 = combine_c (mr2, sr2, da, dr, vl);
                vfloat32m1_t dg2 = combine_c (mg2, sg2, da, dg, vl);
                vfloat32m1_t db2 = combine_c (mb2, sb2, da, db, vl);

                __riscv_vsse32_v_f32m1(&pd[0],stride,da2,vl);
                __riscv_vsse32_v_f32m1(&pd[1],stride,dr2,vl);
                __riscv_vsse32_v_f32m1(&pd[2],stride,dg2,vl);
                __riscv_vsse32_v_f32m1(&pd[3],stride,db2,vl);

                vl_step = vl * component_count;
            }
        }
        else
        {
            for (; vn > 0; vn -= vl_step, pd += vl_step, ps += vl_step, pm += vl_step)
            {
                vl = __riscv_vsetvl_e32m1(vn);

                vfloat32m1_t da = __riscv_vlse32_v_f32m1(&pd[0],stride,vl);
                vfloat32m1_t dr = __riscv_vlse32_v_f32m1(&pd[1],stride,vl);
                vfloat32m1_t dg = __riscv_vlse32_v_f32m1(&pd[2],stride,vl);
                vfloat32m1_t db = __riscv_vlse32_v_f32m1(&pd[3],stride,vl);

                vfloat32m1_t sa = __riscv_vlse32_v_f32m1(&ps[0],stride,vl);
                vfloat32m1_t sr = __riscv_vlse32_v_f32m1(&ps[1],stride,vl);
                vfloat32m1_t sg = __riscv_vlse32_v_f32m1(&ps[2],stride,vl);
                vfloat32m1_t sb = __riscv_vlse32_v_f32m1(&ps[3],stride,vl);

                vfloat32m1_t ma = __riscv_vlse32_v_f32m1(&pm[0],stride,vl);
                vfloat32m1_t mr = __riscv_vlse32_v_f32m1(&pm[0],stride,vl);
                vfloat32m1_t mg = __riscv_vlse32_v_f32m1(&pm[0],stride,vl);
                vfloat32m1_t mb = __riscv_vlse32_v_f32m1(&pm[0],stride,vl);

                vfloat32m1_t sa2 = __riscv_vfmul_vv_f32m1(sa, ma, vl);
                vfloat32m1_t sr2 = __riscv_vfmul_vv_f32m1(sr, ma, vl);
                vfloat32m1_t sg2 = __riscv_vfmul_vv_f32m1(sg, ma, vl);
                vfloat32m1_t sb2 = __riscv_vfmul_vv_f32m1(sb, ma, vl);

                ma = mr = mg = mb = sa2;

                vfloat32m1_t dr2 = combine_c (ma, sr2, da, dr, vl);
                vfloat32m1_t dg2 = combine_c (mr, sg2, da, dg, vl);
                vfloat32m1_t db2 = combine_c (mg, sb2, da, db, vl);
                vfloat32m1_t da2 = combine_a (mb, sa2, da, da, vl);

                __riscv_vsse32_v_f32m1(&pd[0],stride,da2,vl);
                __riscv_vsse32_v_f32m1(&pd[1],stride,dr2,vl);
                __riscv_vsse32_v_f32m1(&pd[2],stride,dg2,vl);
                __riscv_vsse32_v_f32m1(&pd[3],stride,db2,vl);

                vl_step = vl * component_count;
            }
        }
    }
}

#define RVV_MAKE_COMBINER(name, component, combine_a, combine_c)	\
    static void								\
    rvv_combine_ ## name ## _float (pixman_implementation_t *imp,	\
				pixman_op_t              op,		\
				float                   *dest,		\
				const float             *src,		\
				const float             *mask,		\
				int		         n_pixels)	\
    {									\
	rvv_combine_inner (component, dest, src, mask, n_pixels,	\
		       combine_a, combine_c);				\
    }

#define RVV_MAKE_COMBINERS(name, combine_a, combine_c)			\
    RVV_MAKE_COMBINER(name ## _ca, TRUE, combine_a, combine_c)		\
    RVV_MAKE_COMBINER(name ## _u, FALSE, combine_a, combine_c)

static force_inline vfloat32m1_t
rvv_get_factor (combine_factor_t factor, vfloat32m1_t sa, vfloat32m1_t da, size_t vl)
{
    vfloat32m1_t vone = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);

    switch (factor)
    {
    case ZERO:
        return vzero;
	break;

    case ONE:
        return vone;
	break;

    case SRC_ALPHA:
	    return sa;
	break;

    case DEST_ALPHA:
	    return da;
	break;

    case INV_SA:
	    return __riscv_vfsub_vv_f32m1(vone, sa, vl);
	break;

    case INV_DA:
	    return __riscv_vfsub_vv_f32m1(vone, da, vl);
	break;

    case SA_OVER_DA:
        return __riscv_vmerge_vvm_f32m1(vone, __riscv_vfmin_vv_f32m1(vone, __riscv_vfmax_vv_f32m1(vzero, __riscv_vfdiv_vv_f32m1(sa, da, vl), vl), vl), __riscv_vmfne_vf_f32m1_b32(da, 0.0f, vl), vl);

    case DA_OVER_SA:
        return __riscv_vmerge_vvm_f32m1(__riscv_vfmin_vv_f32m1(vone, __riscv_vfmax_vv_f32m1(vzero, __riscv_vfdiv_vv_f32m1(da, sa, vl), vl), vl), vone, __riscv_vmfeq_vf_f32m1_b32(sa, 0.0f, vl), vl);

    case INV_SA_OVER_DA:
    {
        vfloat32m1_t t0 =__riscv_vfdiv_vv_f32m1(__riscv_vfsub_vv_f32m1(vone, sa, vl), da, vl);
        return __riscv_vmerge_vvm_f32m1(vone,
            __riscv_vfmin_vv_f32m1(vone, __riscv_vfmax_vv_f32m1(vzero, t0, vl), vl), __riscv_vmfne_vf_f32m1_b32(da, 0.0f, vl), vl);
    }

    case INV_DA_OVER_SA:
    {
        vfloat32m1_t t0 =__riscv_vfdiv_vv_f32m1(__riscv_vfsub_vv_f32m1(vone, da, vl), sa, vl);
        return __riscv_vmerge_vvm_f32m1(vone,
            __riscv_vfmin_vv_f32m1(vone, __riscv_vfmax_vv_f32m1(vzero, t0, vl), vl), __riscv_vmfne_vf_f32m1_b32(sa, 0.0f, vl), vl);
    }

    case ONE_MINUS_SA_OVER_DA:
    {
        vfloat32m1_t t0 =__riscv_vfsub_vv_f32m1(vone, __riscv_vfdiv_vv_f32m1(sa, da, vl), vl);
        return __riscv_vmerge_vvm_f32m1(vzero,
            __riscv_vfmin_vv_f32m1(vone, __riscv_vfmax_vv_f32m1(vzero, t0, vl), vl), __riscv_vmfne_vf_f32m1_b32(da, 0.0f, vl), vl);
    }

    case ONE_MINUS_DA_OVER_SA:
    {
        vfloat32m1_t t0 =__riscv_vfsub_vv_f32m1(vone, __riscv_vfdiv_vv_f32m1(da, sa, vl), vl);
        return __riscv_vmerge_vvm_f32m1(vzero,
            __riscv_vfmin_vv_f32m1(vone, __riscv_vfmax_vv_f32m1(vzero, t0, vl), vl), __riscv_vmfne_vf_f32m1_b32(sa, 0.0f, vl), vl);
    }

    case ONE_MINUS_INV_DA_OVER_SA:
    {
        vfloat32m1_t t0 = __riscv_vfsub_vv_f32m1(vone, __riscv_vfdiv_vv_f32m1(__riscv_vfsub_vv_f32m1(vone, da, vl), sa, vl), vl);
        return __riscv_vmerge_vvm_f32m1(vzero,
            __riscv_vfmin_vv_f32m1(vone, __riscv_vfmax_vv_f32m1(vzero, t0, vl), vl), __riscv_vmfne_vf_f32m1_b32(sa, 0.0f, vl), vl);
    }

    case ONE_MINUS_INV_SA_OVER_DA:
    {
        vfloat32m1_t t0 = __riscv_vfsub_vv_f32m1(vone, __riscv_vfdiv_vv_f32m1(__riscv_vfsub_vv_f32m1(vone, sa, vl), da, vl), vl);
        return __riscv_vmerge_vvm_f32m1(__riscv_vfmin_vv_f32m1(vone, __riscv_vfmax_vv_f32m1(vzero, t0, vl), vl),
            vzero, __riscv_vmfeq_vf_f32m1_b32(da, 0.0f, vl), vl);
    }
    }

    return __riscv_vfmv_v_f_f32m1(-1.0f, vl);
}

#define RVV_MAKE_PD_COMBINERS(name, a, b)					                                                            \
    static vfloat32m1_t force_inline						                                                            \
    rvv_pd_combine_ ## name (vfloat32m1_t sa, vfloat32m1_t s, vfloat32m1_t da, vfloat32m1_t d, size_t vl)               \
    {                                                                                                                   \
	const vfloat32m1_t fa = rvv_get_factor (a, sa, da, vl);			                                                    \
	const vfloat32m1_t fb = rvv_get_factor (b, sa, da, vl);			                                                    \
	vfloat32m1_t t0 = __riscv_vfadd_vv_f32m1(__riscv_vfmul_vv_f32m1(s,fa,vl),__riscv_vfmul_vv_f32m1(d,fb,vl),vl);	    \
	return __riscv_vmerge_vvm_f32m1(__riscv_vfmv_v_f_f32m1(1.0f, vl), t0,__riscv_vmflt_vf_f32m1_b32(t0, 1.0f, vl), vl); \
    }									                                                                                \
									                                                                                \
    RVV_MAKE_COMBINERS(name, rvv_pd_combine_ ## name, rvv_pd_combine_ ## name)

RVV_MAKE_PD_COMBINERS (clear,			ZERO,				ZERO)
RVV_MAKE_PD_COMBINERS (src,				ONE,				ZERO)
RVV_MAKE_PD_COMBINERS (dst,				ZERO,				ONE)
RVV_MAKE_PD_COMBINERS (over,			ONE,				INV_SA)
RVV_MAKE_PD_COMBINERS (over_reverse,		INV_DA,				ONE)
RVV_MAKE_PD_COMBINERS (in,				DEST_ALPHA,			ZERO)
RVV_MAKE_PD_COMBINERS (in_reverse,			ZERO,				SRC_ALPHA)
RVV_MAKE_PD_COMBINERS (out,				INV_DA,				ZERO)
RVV_MAKE_PD_COMBINERS (out_reverse,			ZERO,				INV_SA)
RVV_MAKE_PD_COMBINERS (atop,			DEST_ALPHA,			INV_SA)
RVV_MAKE_PD_COMBINERS (atop_reverse,		INV_DA,				SRC_ALPHA)
RVV_MAKE_PD_COMBINERS (xor,				INV_DA,				INV_SA)
RVV_MAKE_PD_COMBINERS (add,				ONE,				ONE)

RVV_MAKE_PD_COMBINERS (saturate,			INV_DA_OVER_SA,			ONE)

RVV_MAKE_PD_COMBINERS (disjoint_clear,		ZERO,				ZERO)
RVV_MAKE_PD_COMBINERS (disjoint_src,		ONE,				ZERO)
RVV_MAKE_PD_COMBINERS (disjoint_dst,		ZERO,				ONE)
RVV_MAKE_PD_COMBINERS (disjoint_over,		ONE,				INV_SA_OVER_DA)
RVV_MAKE_PD_COMBINERS (disjoint_over_reverse,	INV_DA_OVER_SA,			ONE)
RVV_MAKE_PD_COMBINERS (disjoint_in,			ONE_MINUS_INV_DA_OVER_SA,	ZERO)
RVV_MAKE_PD_COMBINERS (disjoint_in_reverse,		ZERO,				ONE_MINUS_INV_SA_OVER_DA)
RVV_MAKE_PD_COMBINERS (disjoint_out,		INV_DA_OVER_SA,			ZERO)
RVV_MAKE_PD_COMBINERS (disjoint_out_reverse,	ZERO,				INV_SA_OVER_DA)
RVV_MAKE_PD_COMBINERS (disjoint_atop,		ONE_MINUS_INV_DA_OVER_SA,	INV_SA_OVER_DA)
RVV_MAKE_PD_COMBINERS (disjoint_atop_reverse,	INV_DA_OVER_SA,			ONE_MINUS_INV_SA_OVER_DA)
RVV_MAKE_PD_COMBINERS (disjoint_xor,		INV_DA_OVER_SA,			INV_SA_OVER_DA)

RVV_MAKE_PD_COMBINERS (conjoint_clear,		ZERO,				ZERO)
RVV_MAKE_PD_COMBINERS (conjoint_src,		ONE,				ZERO)
RVV_MAKE_PD_COMBINERS (conjoint_dst,		ZERO,				ONE)
RVV_MAKE_PD_COMBINERS (conjoint_over,		ONE,				ONE_MINUS_SA_OVER_DA)
RVV_MAKE_PD_COMBINERS (conjoint_over_reverse,	ONE_MINUS_DA_OVER_SA,		ONE)
RVV_MAKE_PD_COMBINERS (conjoint_in,			DA_OVER_SA,			ZERO)
RVV_MAKE_PD_COMBINERS (conjoint_in_reverse,		ZERO,				SA_OVER_DA)
RVV_MAKE_PD_COMBINERS (conjoint_out,		ONE_MINUS_DA_OVER_SA,		ZERO)
RVV_MAKE_PD_COMBINERS (conjoint_out_reverse,	ZERO,				ONE_MINUS_SA_OVER_DA)
RVV_MAKE_PD_COMBINERS (conjoint_atop,		DA_OVER_SA,			ONE_MINUS_SA_OVER_DA)
RVV_MAKE_PD_COMBINERS (conjoint_atop_reverse,	ONE_MINUS_DA_OVER_SA,		SA_OVER_DA)
RVV_MAKE_PD_COMBINERS (conjoint_xor,		ONE_MINUS_DA_OVER_SA,		ONE_MINUS_SA_OVER_DA)

#define RVV_MAKE_SEPARABLE_PDF_COMBINERS(name)				        \
    static force_inline vfloat32m1_t						\
    rvv_combine_ ## name ## _a (vfloat32m1_t sa, vfloat32m1_t s, vfloat32m1_t da, vfloat32m1_t d, size_t vl) \
    {									        \
       return __riscv_vfsub_vv_f32m1(__riscv_vfadd_vv_f32m1(da,sa,vl), __riscv_vfmul_vv_f32m1(da,sa,vl),vl); \
    }									        \
									        \
    static force_inline vfloat32m1_t						                             \
    rvv_combine_ ## name ## _c (vfloat32m1_t sa, vfloat32m1_t s, vfloat32m1_t da, vfloat32m1_t d, size_t vl) \
    {									                                     \
        vfloat32m1_t f = __riscv_vfmul_vf_f32m1(                                                             \
                __riscv_vfadd_vv_f32m1(                                                                      \
                    __riscv_vfmul_vv_f32m1( __riscv_vfsub_vf_f32m1(sa, 1.0f, vl), d, vl)                 \
                    , __riscv_vfmul_vv_f32m1(__riscv_vfsub_vf_f32m1(da, 1.0f, vl), s, vl)                \
                    ,vl)                                                                                 \
                ,-1.0f, vl);                                                                                 \
									                                     \
	return __riscv_vfadd_vv_f32m1(f, rvv_blend_ ## name (sa, s, da, d, vl), vl);			     \
    }									                                     \
									                                     \
    RVV_MAKE_COMBINERS (name, rvv_combine_ ## name ## _a, rvv_combine_ ## name ## _c)

RVV_MAKE_SEPARABLE_PDF_COMBINERS (multiply)
RVV_MAKE_SEPARABLE_PDF_COMBINERS (screen)
RVV_MAKE_SEPARABLE_PDF_COMBINERS (overlay)
RVV_MAKE_SEPARABLE_PDF_COMBINERS (darken)
RVV_MAKE_SEPARABLE_PDF_COMBINERS (lighten)
RVV_MAKE_SEPARABLE_PDF_COMBINERS (color_dodge)
RVV_MAKE_SEPARABLE_PDF_COMBINERS (color_burn)
RVV_MAKE_SEPARABLE_PDF_COMBINERS (hard_light)
RVV_MAKE_SEPARABLE_PDF_COMBINERS (soft_light)
RVV_MAKE_SEPARABLE_PDF_COMBINERS (difference)
RVV_MAKE_SEPARABLE_PDF_COMBINERS (exclusion)



/*
 * PDF nonseperable blend modes are implemented using the following functions
 * to operate in Hsl space, with Cmax, Cmid, Cmin referring to the max, mid
 * and min value of the red, green and blue components.
 *
 * LUM (C) = 0.3 × Cred + 0.59 × Cgreen + 0.11 × Cblue
 *
 * clip_color (C):
 *     l = LUM (C)
 *     min = Cmin
 *     max = Cmax
 *     if n < 0.0
 *         C = l + (((C – l) × l) ⁄ (l – min))
 *     if x > 1.0
 *         C = l + (((C – l) × (1 – l) ) ⁄ (max – l))
 *     return C
 *
 * set_lum (C, l):
 *     d = l – LUM (C)
 *     C += d
 *     return clip_color (C)
 *
 * SAT (C) = CH_MAX (C) - CH_MIN (C)
 *
 * set_sat (C, s):
 *     if Cmax > Cmin
 *         Cmid = ( ( ( Cmid – Cmin ) × s ) ⁄ ( Cmax – Cmin ) )
 *         Cmax = s
 *     else
 *         Cmid = Cmax = 0.0
 *         Cmin = 0.0
 *     return C
 */

/* For premultiplied colors, we need to know what happens when C is
 * multiplied by a real number. LUM and SAT are linear:
 *
 *     LUM (r × C) = r × LUM (C)	SAT (r * C) = r * SAT (C)
 *
 * If we extend clip_color with an extra argument a and change
 *
 *     if x >= 1.0
 *
 * into
 *
 *     if x >= a
 *
 * then clip_color is also linear:
 *
 *     r * clip_color (C, a) = clip_color (r * C, r * a);
 *
 * for positive r.
 *
 * Similarly, we can extend set_lum with an extra argument that is just passed
 * on to clip_color:
 *
 *       r * set_lum (C, l, a)
 *
 *     = r × clip_color (C + l - LUM (C), a)
 *
 *     = clip_color (r * C + r × l - r * LUM (C), r * a)
 *
 *     = set_lum (r * C, r * l, r * a)
 *
 * Finally, set_sat:
 *
 *       r * set_sat (C, s) = set_sat (x * C, r * s)
 *
 * The above holds for all non-zero x, because the x'es in the fraction for
 * C_mid cancel out. Specifically, it holds for x = r:
 *
 *       r * set_sat (C, s) = set_sat (r * C, r * s)
 *
 */

__attribute__((always_inline)) inline static vfloat32m1_t
rvv_minf(const vfloat32m1_t a, const vfloat32m1_t b, const size_t vl)
{
    register vbool32_t vlt;
    vlt = __riscv_vmflt_vv_f32m1_b32(a, b, vl);
    return __riscv_vmerge_vvm_f32m1(b, a, vlt, vl);
}

__attribute__((always_inline)) inline static vfloat32m1_t
rvv_maxf(const vfloat32m1_t a, const vfloat32m1_t b, const size_t vl)
{
    register vbool32_t vgt;
    vgt = __riscv_vmfgt_vv_f32m1_b32(a, b, vl);
    return __riscv_vmerge_vvm_f32m1(b, a, vgt, vl);
}

__attribute__((always_inline)) inline static vfloat32m1_t
rvv_channel_min(const vfloat32m1_t r, const vfloat32m1_t g,
                const vfloat32m1_t b, const size_t vl)
{
    register vfloat32m1_t vret;
    vret = rvv_minf(rvv_minf(r, g, vl), b, vl);
    return (vret);
}

__attribute__((always_inline)) inline static vfloat32m1_t
rvv_channel_max(const vfloat32m1_t r, const vfloat32m1_t g,
                const vfloat32m1_t b, const size_t vl)
{
    register vfloat32m1_t vret;
    vret = rvv_maxf(rvv_maxf(r, g, vl), b, vl);
    return (vret);
}

__attribute__((always_inline)) inline static vfloat32m1_t
rvv_get_lum(const vfloat32m1_t r, const vfloat32m1_t g, const vfloat32m1_t b,
            const size_t vl)
{
    register vfloat32m1_t vret;
    vret = __riscv_vfmadd_vf_f32m1(
        r, 0.3f,
        __riscv_vfmadd_vf_f32m1(g, 0.59f, __riscv_vfmul_vf_f32m1(b, 0.11f, vl),
                                vl),
        vl);
    return (vret);
}

__attribute__((always_inline)) inline static vfloat32m1_t
rvv_get_sat(const vfloat32m1_t r, const vfloat32m1_t g, const vfloat32m1_t b,
            const size_t vl)
{
    register vfloat32m1_t vret;
    vret = __riscv_vfsub_vv_f32m1(rvv_channel_max(r, g, b, vl),
                                  rvv_channel_min(r, g, b, vl), vl);
    return (vret);
}

__attribute__((always_inline))
inline static void rvv_clip_color(vfloat32m1_t *__restrict__ r, vfloat32m1_t *__restrict__ g,
               vfloat32m1_t *__restrict__ b, const vfloat32m1_t a,
               const size_t vl)
{
    const vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, vl);
    register vfloat32m1_t vr, vg, vb;
    register vfloat32m1_t t0, t1, t2;
    register vfloat32m1_t asl;
    vfloat32m1_t t;
    vfloat32m1_t invt;
    vbool32_t n_lt_0;
    vbool32_t x_gt_a;
    vbool32_t t_lt_fltmin;
    vr = *(r);
    vg = *(g);
    vb = *(b);
    register vfloat32m1_t l = rvv_get_lum(vr, vg, vb, vl);
    register vfloat32m1_t n = rvv_channel_min(vr, vg, vb, vl);
    n_lt_0 = __riscv_vmflt_vf_f32m1_b32(n, 0.0f, vl);
    register vfloat32m1_t x = rvv_channel_max(vr, vg, vb, vl);
    if (__riscv_vfirst_m_b32(n_lt_0, vl))
    {
        t = __riscv_vfsub_vv_f32m1(l, n, vl);
        t_lt_fltmin = __riscv_vmflt_vf_f32m1_b32(t, FLT_MIN, vl);
        invt = __riscv_vfrec7_v_f32m1(t, vl);
        t0 = __riscv_vfmul_vv_f32m1(
            __riscv_vfmul_vv_f32m1(__riscv_vfsub_vv_f32m1(vr, l, vl), l, vl),
            invt, vl);
        *(r) = __riscv_vmerge_vvm_f32m1(__riscv_vfadd_vv_f32m1(l, t0, vl),
                                        vzero, t_lt_fltmin, vl);
        t1 = __riscv_vfmul_vv_f32m1(
            __riscv_vfmul_vv_f32m1(__riscv_vfsub_vv_f32m1(vg, l, vl), l, vl),
            invt, vl);
        *(g) = __riscv_vmerge_vvm_f32m1(__riscv_vfadd_vv_f32m1(l, t1, vl),
                                        vzero, t_lt_fltmin, vl);
        t2 = __riscv_vfmul_vv_f32m1(
            __riscv_vfmul_vv_f32m1(__riscv_vfsub_vv_f32m1(vb, l, vl), l, vl),
            invt, vl);
        *(b) = __riscv_vmerge_vvm_f32m1(__riscv_vfadd_vv_f32m1(l, t2, vl),
                                        vzero, t_lt_fltmin, vl);
    }
    x_gt_a = __riscv_vmfgt_vv_f32m1_b32(x, a, vl);
    if (__riscv_vfirst_m_b32(x_gt_a, vl))
    {
        t = __riscv_vfsub_vv_f32m1(x, l, vl);
        t_lt_fltmin = __riscv_vmflt_vf_f32m1_b32(t, FLT_MIN, vl);
        asl = __riscv_vfsub_vv_f32m1(a, l, vl);
        invt = __riscv_vfrec7_v_f32m1(t, vl);
        t0 = __riscv_vfmul_vv_f32m1(
            __riscv_vfmul_vv_f32m1(__riscv_vfsub_vv_f32m1(vr, l, vl), asl, vl),
            invt, vl);
        *(r) = __riscv_vmerge_vvm_f32m1(__riscv_vfadd_vv_f32m1(l, t0, vl), a,
                                        t_lt_fltmin, vl);
        t1 = __riscv_vfmul_vv_f32m1(
            __riscv_vfmul_vv_f32m1(__riscv_vfsub_vv_f32m1(vg, l, vl), asl, vl),
            invt, vl);
        *(g) = __riscv_vmerge_vvm_f32m1(__riscv_vfadd_vv_f32m1(l, t1, vl), a,
                                        t_lt_fltmin, vl);
        t2 = __riscv_vfmul_vv_f32m1(
            __riscv_vfmul_vv_f32m1(__riscv_vfsub_vv_f32m1(vb, l, vl), asl, vl),
            invt, vl);
        *(b) = __riscv_vmerge_vvm_f32m1(__riscv_vfadd_vv_f32m1(l, t2, vl), a,
                                        t_lt_fltmin, vl);
    }
}

__attribute__((always_inline))
inline static void rvv_set_lum(vfloat32m1_t *__restrict__ r, vfloat32m1_t *__restrict__ g,
            vfloat32m1_t *__restrict__ b, const vfloat32m1_t sa,
            const vfloat32m1_t l, const size_t vl)
{
    register vfloat32m1_t d;
    register vfloat32m1_t vr, vg, vb;
    vr = *(r);
    vg = *(g);
    vb = *(b);
    d = __riscv_vfsub_vv_f32m1(l, rvv_get_lum(vr, vg, vb, vl), vl);
    *(r) = __riscv_vfadd_vv_f32m1(vr, d, vl);
    *(g) = __riscv_vfadd_vv_f32m1(vg, d, vl);
    *(b) = __riscv_vfadd_vv_f32m1(vb, d, vl);
    rvv_clip_color(r, g, b, sa, vl);
}

__attribute__((always_inline))
inline static void rvv_set_sat(vfloat32m1_t *__restrict__ r, vfloat32m1_t *__restrict__ g,
            vfloat32m1_t *__restrict__ b, const vfloat32m1_t sat,
            const size_t vl)
{
    const vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, vl);
    register vfloat32m1_t vr, vg, vb;
    register vfloat32m1_t t, t0;
    register vbool32_t r_gt_g;
    register vbool32_t g_gt_b;
    register vbool32_t r_gt_b;
    register vbool32_t t_lt_fltmin;
    vfloat32m1_t *__restrict__ max;
    vfloat32m1_t *__restrict__ mid;
    vfloat32m1_t *__restrict__ min;

    vr = *(r);
    vg = *(g);
    vb = *(b);
    r_gt_g = __riscv_vmfgt_vv_f32m1_b32(vr, vg, vl);
    r_gt_b = __riscv_vmfgt_vv_f32m1_b32(vr, vb, vl);
    if (__riscv_vfirst_m_b32(r_gt_g, vl))
    {
        if (__riscv_vfirst_m_b32(r_gt_b, vl))
        {
            max = r;
            g_gt_b = __riscv_vmfgt_vv_f32m1_b32(vg, vb, vl);
            if (__riscv_vfirst_m_b32(g_gt_b, vl))
            {
                mid = g;
                min = b;
            }
            else
            {
                mid = b;
                min = g;
            }
        }
        else
        {
            max = b;
            mid = r;
            min = g;
        }
    }
    else
    {
        if (__riscv_vfirst_m_b32(r_gt_b, vl))
        {
            max = g;
            mid = r;
            min = b;
        }
        else
        {
            min = r;
            g_gt_b = __riscv_vmfgt_vv_f32m1_b32(vg, vb, vl);
            if (__riscv_vfirst_m_b32(g_gt_b, vl))
            {
                max = g;
                mid = b;
            }
            else
            {
                max = b;
                mid = g;
            }
        }
    }

    t = __riscv_vfsub_vv_f32m1(*(max), *(min), vl);
    t_lt_fltmin = __riscv_vmflt_vf_f32m1_b32(t, FLT_MIN, vl);
    t0 = __riscv_vfmul_vv_f32m1(__riscv_vfsub_vv_f32m1(*(mid), *(min), vl), sat,
                                vl);
    *(mid) = __riscv_vmerge_vvm_f32m1(__riscv_vfdiv_vv_f32m1(t0, t, vl), vzero,
                                      t_lt_fltmin, vl);
    *(max) = __riscv_vmerge_vvm_f32m1(sat, vzero, t_lt_fltmin, vl);
    *(min) = vzero;
}

/* Hue:
 *
 *       as * ad * B(s/as, d/as)
 *     = as * ad * set_lum (set_sat (s/as, SAT (d/ad)), LUM (d/ad), 1)
 *     = set_lum (set_sat (ad * s, as * SAT (d)), as * LUM (d), as * ad)
 *
 */
__attribute__((always_inline))
inline static void rvv_blend_hsl_hue(
    vfloat32m1_t *__restrict__ res_r, vfloat32m1_t *__restrict__ res_g,
    vfloat32m1_t *__restrict__ res_b, const vfloat32m1_t *__restrict__ dst_r,
    const vfloat32m1_t *__restrict__ dst_g,
    const vfloat32m1_t *__restrict__ dst_b, const vfloat32m1_t da,
    vfloat32m1_t *__restrict__ src_r, vfloat32m1_t *__restrict__ src_g,
    vfloat32m1_t *__restrict__ src_b, const vfloat32m1_t sa, const size_t vl)
{
    register vfloat32m1_t vr, vg, vb;
    vr = *(src_r);
    *(res_r) = __riscv_vfmul_vv_f32m1(vr, da, vl);
    vg = *(src_g);
    *(res_g) = __riscv_vfmul_vv_f32m1(vg, da, vl);
    vb = *(src_b);
    *(res_b) = __riscv_vfmul_vv_f32m1(vb, da, vl);
    rvv_set_sat(res_r, res_g, res_b,
                __riscv_vfmul_vv_f32m1(
                    rvv_get_sat(*(dst_r), *(dst_g), *(dst_b), vl), sa, vl),
                vl);
    rvv_set_lum(res_r, res_g, res_b, __riscv_vfmul_vv_f32m1(sa, da, vl),
                __riscv_vfmul_vv_f32m1(
                    rvv_get_lum(*(dst_r), *(dst_g), *(dst_b), vl), sa, vl),
                vl);
}

/*
 * Saturation
 *
 *     as * ad * B(s/as, d/ad)
 *   = as * ad * set_lum (set_sat (d/ad, SAT (s/as)), LUM (d/ad), 1)
 *   = set_lum (as * ad * set_sat (d/ad, SAT (s/as)),
 *                                       as * LUM (d), as * ad)
 *   = set_lum (set_sat (as * d, ad * SAT (s), as * LUM (d), as * ad))
 */
__attribute__((always_inline))
inline static void rvv_blend_hsl_saturation(
    vfloat32m1_t *__restrict__ res_r, vfloat32m1_t *__restrict__ res_g,
    vfloat32m1_t *__restrict__ res_b, const vfloat32m1_t *__restrict__ dst_r,
    const vfloat32m1_t *__restrict__ dst_g,
    const vfloat32m1_t *__restrict__ dst_b, const vfloat32m1_t da,
    vfloat32m1_t *__restrict__ src_r, vfloat32m1_t *__restrict__ src_g,
    vfloat32m1_t *__restrict__ src_b, const vfloat32m1_t sa, const size_t vl)
{
    register vfloat32m1_t vr, vg, vb;
    vr = *(dst_r);
    *(res_r) = __riscv_vfmul_vv_f32m1(vr, sa, vl);
    vg = *(dst_g);
    *(res_g) = __riscv_vfmul_vv_f32m1(vg, sa, vl);
    vb = *(dst_b);
    *(res_b) = __riscv_vfmul_vv_f32m1(vb, sa, vl);
    rvv_set_sat(res_r, res_g, res_b,
                __riscv_vfmul_vv_f32m1(
                    rvv_get_sat(*(src_r), *(src_g), *(src_b), vl), da, vl),
                vl);
    rvv_set_lum(res_r, res_g, res_b, __riscv_vfmul_vv_f32m1(sa, da, vl),
                __riscv_vfmul_vv_f32m1(rvv_get_lum(vr, vg, vb, vl), sa, vl),
                vl);
}

/*
 * Color
 *
 *     as * ad * B(s/as, d/as)
 *   = as * ad * set_lum (s/as, LUM (d/ad), 1)
 *   = set_lum (s * ad, as * LUM (d), as * ad)
 */
__attribute__((always_inline))
inline static void rvv_blend_hsl_color(
    vfloat32m1_t *__restrict__ res_r, vfloat32m1_t *__restrict__ res_g,
    vfloat32m1_t *__restrict__ res_b, const vfloat32m1_t *__restrict__ dst_r,
    const vfloat32m1_t *__restrict__ dst_g,
    const vfloat32m1_t *__restrict__ dst_b, const vfloat32m1_t da,
    vfloat32m1_t *__restrict__ src_r, vfloat32m1_t *__restrict__ src_g,
    vfloat32m1_t *__restrict__ src_b, const vfloat32m1_t sa, const size_t vl)
{
    register vfloat32m1_t vr, vg, vb;
    vr = *(src_r);
    *(res_r) = __riscv_vfmul_vv_f32m1(vr, da, vl);
    vg = *(src_g);
    *(res_g) = __riscv_vfmul_vv_f32m1(vg, da, vl);
    vb = *(src_b);
    *(res_b) = __riscv_vfmul_vv_f32m1(vb, da, vl);
    rvv_set_lum(res_r, res_g, res_b, __riscv_vfmul_vv_f32m1(sa, da, vl),
                __riscv_vfmul_vv_f32m1(
                    rvv_get_lum(*(dst_r), *(dst_g), *(dst_b), vl), sa, vl),
                vl);
}

/*
 * Luminosity
 *
 *     as * ad * B(s/as, d/ad)
 *   = as * ad * set_lum (d/ad, LUM (s/as), 1)
 *   = set_lum (as * d, ad * LUM (s), as * ad)
 */
__attribute__((always_inline))
inline static void rvv_blend_hsl_luminosity(
    vfloat32m1_t *__restrict__ res_r, vfloat32m1_t *__restrict__ res_g,
    vfloat32m1_t *__restrict__ res_b, const vfloat32m1_t *__restrict__ dst_r,
    const vfloat32m1_t *__restrict__ dst_g,
    const vfloat32m1_t *__restrict__ dst_b, const vfloat32m1_t da,
    vfloat32m1_t *__restrict__ src_r, vfloat32m1_t *__restrict__ src_g,
    vfloat32m1_t *__restrict__ src_b, const vfloat32m1_t sa, const size_t vl)
{
    register vfloat32m1_t vr, vg, vb;
    vr = *(dst_r);
    *(res_r) = __riscv_vfmul_vv_f32m1(vr, sa, vl);
    vg = *(dst_g);
    *(res_g) = __riscv_vfmul_vv_f32m1(vg, sa, vl);
    vb = *(dst_b);
    *(res_b) = __riscv_vfmul_vv_f32m1(vb, sa, vl);
    rvv_set_lum(res_r, res_g, res_b, __riscv_vfmul_vv_f32m1(sa, da, vl),
                __riscv_vfmul_vv_f32m1(
                    rvv_get_lum(*(src_r), *(src_g), *(src_b), vl), da, vl),
                vl);
}

#define RVV_MAKE_NON_SEPARABLE_PDF_COMBINERS(name)                             \
    static void rvv_combine_##name##_u_float(                                  \
        pixman_implementation_t *imp, pixman_op_t op, float *dest,             \
        const float *src, const float *mask, int n_pixels)                     \
    {                                                                          \
        const vfloat32m1_t vone =                                              \
            __riscv_vfmv_v_f_f32m1(1.0f, __riscv_vsetvlmax_e32m1());           \
        float *__restrict__ pd = dest;                                         \
        const float *__restrict__ ps = src;                                    \
        const float *__restrict__ pm = mask;                                   \
        const size_t component_count = 4;                                      \
        size_t vn = component_count * (size_t)n_pixels;                        \
        size_t vl = 0ULL;                                                      \
        size_t vl_step = 0ULL;                                                 \
        const ptrdiff_t stride = 4 * sizeof(float);                            \
        if (mask)                                                              \
        {                                                                      \
            for (; vn > 0ULL;                                                  \
                 vn -= vl_step, pd += vl_step, ps += vl_step, pm += vl_step)   \
            {                                                                  \
                vfloat32m1_t vrr;                                              \
                vfloat32m1_t vrg;                                              \
                vfloat32m1_t vrb;                                              \
                register vfloat32m1_t vma;                                     \
                register vfloat32m1_t vsa;                                     \
                vfloat32m1_t vsr;                                              \
                vfloat32m1_t vsg;                                              \
                vfloat32m1_t vsb;                                              \
                register vfloat32m1_t vda;                                     \
                vfloat32m1_t vdr;                                              \
                vfloat32m1_t vdg;                                              \
                vfloat32m1_t vdb;                                              \
                register vfloat32m1_t t0;                                      \
                register vfloat32m1_t t1;                                      \
                register vfloat32m1_t t2;                                      \
                vl = __riscv_vsetvl_e32m1(vn);                                 \
                __builtin_prefetch(ps + vl_step, 0, 1);                        \
                __builtin_prefetch(pm + vl, 0, 1);                             \
                __builtin_prefetch(pd + vl_step, 0, 1);                        \
                vma = __riscv_vle32_v_f32m1(&pm[0], vl);                       \
                vsa = __riscv_vlse32_v_f32m1(&ps[0], stride, vl);              \
                vda = __riscv_vlse32_v_f32m1(&pd[0], stride, vl);              \
                vsa = __riscv_vfmul_vv_f32m1(vsa, vma, vl);                    \
                vsr = __riscv_vlse32_v_f32m1(&ps[1], stride, vl);              \
                t0 = __riscv_vfsub_vv_f32m1(                                   \
                    __riscv_vfadd_vv_f32m1(vsa, vda, vl),                      \
                    __riscv_vfmul_vv_f32m1(vsa, vda, vl), vl);                 \
                vdr = __riscv_vlse32_v_f32m1(&pd[1], stride, vl);              \
                vsr = __riscv_vfmul_vv_f32m1(vsr, vma, vl);                    \
                vsg = __riscv_vlse32_v_f32m1(&ps[2], stride, vl);              \
                t1 = __riscv_vfsub_vv_f32m1(vone, vsa, vl);                    \
                vdg = __riscv_vlse32_v_f32m1(&pd[2], stride, vl);              \
                vsg = __riscv_vfmul_vv_f32m1(vsg, vma, vl);                    \
                vsb = __riscv_vlse32_v_f32m1(&ps[3], stride, vl);              \
                t2 = __riscv_vfsub_vv_f32m1(vone, vda, vl);                    \
                vdb = __riscv_vlse32_v_f32m1(&pd[3], stride, vl);              \
                vsb = __riscv_vfmul_vv_f32m1(vsb, vma, vl);                    \
                rvv_blend_##name(&vrr, &vrg, &vrb, &vdr, &vdg, &vdb, vda,      \
                                 &vsr, &vsg, &vsb, vsa, vl);                   \
                __riscv_vsse32_v_f32m1(&pd[0], stride, t0, vl);                \
                __riscv_vsse32_v_f32m1(                                        \
                    &pd[1], stride,                                            \
                    __riscv_vfmadd_vv_f32m1(                                   \
                        t1, vdr, __riscv_vfmadd_vv_f32m1(t2, vsr, vrr, vl),    \
                        vl),                                                   \
                    vl);                                                       \
                __riscv_vsse32_v_f32m1(                                        \
                    &pd[2], stride,                                            \
                    __riscv_vfmadd_vv_f32m1(                                   \
                        t1, vdg, __riscv_vfmadd_vv_f32m1(t2, vsg, vrg, vl),    \
                        vl),                                                   \
                    vl);                                                       \
                __riscv_vsse32_v_f32m1(                                        \
                    &pd[3], stride,                                            \
                    __riscv_vfmadd_vv_f32m1(                                   \
                        t1, vdb, __riscv_vfmadd_vv_f32m1(t2, vsb, vrb, vl),    \
                        vl),                                                   \
                    vl);                                                       \
                vl_step = vl * component_count;                                \
            }                                                                  \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            for (; vn > 0ULL; vn -= vl_step, pd += vl_step, ps += vl_step)     \
            {                                                                  \
                vfloat32m1_t vrr;                                              \
                vfloat32m1_t vrg;                                              \
                vfloat32m1_t vrb;                                              \
                register vfloat32m1_t vsa;                                     \
                vfloat32m1_t vsr;                                              \
                vfloat32m1_t vsg;                                              \
                vfloat32m1_t vsb;                                              \
                register vfloat32m1_t vda;                                     \
                vfloat32m1_t vdr;                                              \
                vfloat32m1_t vdg;                                              \
                vfloat32m1_t vdb;                                              \
                register vfloat32m1_t t0;                                      \
                register vfloat32m1_t t1;                                      \
                register vfloat32m1_t t2;                                      \
                vl = __riscv_vsetvl_e32m1(vn);                                 \
                __builtin_prefetch(ps + vl_step, 0, 1);                        \
                __builtin_prefetch(pd + vl_step, 0, 1);                        \
                vsa = __riscv_vlse32_v_f32m1(&ps[0], stride, vl);              \
                vda = __riscv_vlse32_v_f32m1(&pd[0], stride, vl);              \
                vsr = __riscv_vlse32_v_f32m1(&ps[1], stride, vl);              \
                t0 = __riscv_vfsub_vv_f32m1(                                   \
                    __riscv_vfadd_vv_f32m1(vsa, vda, vl),                      \
                    __riscv_vfmul_vv_f32m1(vsa, vda, vl), vl);                 \
                vdr = __riscv_vlse32_v_f32m1(&pd[1], stride, vl);              \
                vsg = __riscv_vlse32_v_f32m1(&ps[2], stride, vl);              \
                t1 = __riscv_vfsub_vv_f32m1(vone, vsa, vl);                    \
                vdg = __riscv_vlse32_v_f32m1(&pd[2], stride, vl);              \
                vsb = __riscv_vlse32_v_f32m1(&ps[3], stride, vl);              \
                t2 = __riscv_vfsub_vv_f32m1(vone, vda, vl);                    \
                vdb = __riscv_vlse32_v_f32m1(&pd[3], stride, vl);              \
                rvv_blend_##name(&vrr, &vrg, &vrb, &vdr, &vdg, &vdb, vda,      \
                                 &vsr, &vsg, &vsb, vsa, vl);                   \
                __riscv_vsse32_v_f32m1(&pd[0], stride, t0, vl);                \
                __riscv_vsse32_v_f32m1(                                        \
                    &pd[1], stride,                                            \
                    __riscv_vfmadd_vv_f32m1(                                   \
                        t1, vdr, __riscv_vfmadd_vv_f32m1(t2, vsr, vrr, vl),    \
                        vl),                                                   \
                    vl);                                                       \
                __riscv_vsse32_v_f32m1(                                        \
                    &pd[2], stride,                                            \
                    __riscv_vfmadd_vv_f32m1(                                   \
                        t1, vdg, __riscv_vfmadd_vv_f32m1(t2, vsg, vrg, vl),    \
                        vl),                                                   \
                    vl);                                                       \
                __riscv_vsse32_v_f32m1(                                        \
                    &pd[3], stride,                                            \
                    __riscv_vfmadd_vv_f32m1(                                   \
                        t1, vdb, __riscv_vfmadd_vv_f32m1(t2, vsb, vrb, vl),    \
                        vl),                                                   \
                    vl);                                                       \
                vl_step = vl * component_count;                                \
            }                                                                  \
        }                                                                      \
    }

RVV_MAKE_NON_SEPARABLE_PDF_COMBINERS(hsl_hue)
RVV_MAKE_NON_SEPARABLE_PDF_COMBINERS(hsl_saturation)
RVV_MAKE_NON_SEPARABLE_PDF_COMBINERS(hsl_color)
RVV_MAKE_NON_SEPARABLE_PDF_COMBINERS(hsl_luminosity)

#endif /*__PIXMAN_RVV_BLENDINGS_H__*/
