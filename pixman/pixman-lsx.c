/*
 * Copyright © 2023 Loongson Technology Corporation Limited
 * Contributed by Ding Song(songding@loongson.cn)
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
 */

#ifdef HAVE_CONFIG_H
#include <pixman-config.h>
#endif

#include "pixman-private.h"
#include "pixman-combine32.h"
#include "loongson_intrinsics.h"

static force_inline uint32_t
over (uint32_t src, uint32_t dest)
{
    uint32_t a = ~src >> 24;

    UN8x4_MUL_UN8_ADD_UN8x4(dest, a, src);

    return dest;
}

static force_inline uint32_t
in (uint32_t x, uint8_t y)
{
    uint16_t a = y;

    UN8x4_MUL_UN8(x, a);

    return x;
}

static force_inline uint32_t
combine_mask (const uint32_t *src, const uint32_t *mask, int i)
{
    uint32_t s, m;

    if (mask) {
        m = *(mask + i) >> A_SHIFT;
        if (!m)
            return 0;
    }
    s = *(src + i);
    if (mask)
       UN8x4_MUL_UN8(s, m);
    return s;
}

static void
combine_mask_ca (uint32_t *src, uint32_t *mask)
{
    uint32_t a = *mask;
    uint32_t x;
    uint16_t xa;

    if (!a) {
        *(src) = 0;
        return;
    }

    x = *(src);
    if (a == ~0) {
        x = x >> A_SHIFT;
        x |= x << G_SHIFT;
        x |= x << R_SHIFT;
        *(mask) = x;
        return;
    }
    xa = x >> A_SHIFT;
    UN8x4_MUL_UN8x4(x, a);
    *(src) = x;

    UN8x4_MUL_UN8(a, xa);
    *(mask) = a;
}

static void
combine_mask_value_ca (uint32_t *src, const uint32_t *mask)
{
    uint32_t a = *mask;
    uint32_t x;

    if (!a) {
        *(src) = 0;
        return;
    }

    if (a == ~0)
        return;

    x = *(src);
    UN8x4_MUL_UN8x4(x, a);
    *(src) = x;
}

static void
combine_mask_alpha_ca (const uint32_t *src, uint32_t *mask)
{
    uint32_t a = *(mask);
    uint32_t x;

    if (!a)
        return;
    x = *(src) >> A_SHIFT;

    if (x == MASK)
        return;

    if (a == -1) {
        x |= x << G_SHIFT;
        x |= x << R_SHIFT;
        *(mask) = x;
        return;
    }
    UN8x4_MUL_UN8(a, x);
    *(mask) = a;
}

/* Compute the product of two unsigned fixed-point 8-bit values from 0 to 1
 * and map its result to the same range.
 *
 * Jim Blinn gives multiple ways to compute this in "Jim Blinn's Corner:
 * Notation, Notation, Notation", the first of which is
 *
 *   prod(a, b) = (a * b + 128) / 255.
 *
 * By approximating the division by 255 as 257/65536, it can be replaced by a
 * multiply and a right shift. This is the implementation that we use in
 * pix_multiply(), but we _mm_mulhi_pu16() by 257 (part of SSE1 or Extended
 * 3DNow!, and unavailable at the time of the book's publication) to perform
 * the multiplication and right shift in a single operation.
 *
 *   prod(a, b) = ((a * b + 128) * 257) >> 16.
 *
 * A third way (how pix_multiply() was implemented prior to 14208344) exists
 * also that performs the multiplication by 257 with adds and shifts.
 *
 * Where temp = a * b + 128
 *
 *   prod(a, b) = (temp + (temp >> 8)) >> 8.
 *
 * The lsx_pix_multiply(src, mask) implemented with the third way, and calculates
 * two sets of data each time.
 */

static force_inline __m128i
lsx_pix_multiply (__m128i src, __m128i mask)
{
    __m128i tmp0, tmp1;
    __m128i vec;

    vec  = __lsx_vreplgr2vr_h(0x80);
    tmp0 = __lsx_vmadd_h(vec, src, mask);
    tmp1 = __lsx_vsrli_h(tmp0, 8);
    tmp0 = __lsx_vadd_h(tmp0, tmp1);
    tmp1 = __lsx_vsrli_h(tmp0, 8);

    return tmp1;
}

static force_inline __m128i
over_1x128 (__m128i src, __m128i alpha, __m128i dst)
{
    __m128i mask_00ff = __lsx_vreplgr2vr_h(0x00ff);

    alpha = __lsx_vxor_v(alpha, mask_00ff);
    alpha = lsx_pix_multiply(dst, alpha);

    return (__lsx_vsadd_bu(src, alpha));
}

static force_inline uint32_t
core_combine_over_u32 (uint32_t src, uint32_t dst)
{
    uint8_t a = src >> 24;

    if (a == 0xff) {
        return src;
    }
    else if (src) {
        __m128i zero = __lsx_vldi(0);
        __m128i vr_src = __lsx_vinsgr2vr_w(zero, src, 0);
        __m128i vr_dst = __lsx_vinsgr2vr_w(zero, dst, 0);
        __m128i vr_alpha;
        __m128i tmp;

        vr_src = __lsx_vilvl_b(zero, vr_src);
        vr_dst = __lsx_vilvl_b(zero, vr_dst);
        vr_alpha = __lsx_vshuf4i_h(vr_src, 0xff);

        tmp = __lsx_vpickev_b(zero, over_1x128(vr_src, vr_alpha, vr_dst));

        return __lsx_vpickve2gr_wu(tmp, 0);
    }

    return dst;
}

static force_inline __m128i
lsx_over_u (__m128i src, __m128i dest)
{
    __m128i r1, r2, r3, t;
    __m128i rb_mask          = __lsx_vreplgr2vr_w(0xff00ff);
    __m128i rb_one_half      = __lsx_vreplgr2vr_w(0x800080);
    __m128i rb_mask_plus_one = __lsx_vreplgr2vr_w(0x10000100);
    __m128i a                = __lsx_vsrli_w(__lsx_vnor_v(src, src), 24);

    r1 = __lsx_vand_v(dest, rb_mask);
    r1 = __lsx_vmadd_w(rb_one_half, r1, a);
    t  = __lsx_vand_v(rb_mask, __lsx_vsrli_w(r1, 8));
    r1 = __lsx_vadd_w(r1, t);
    r1 = __lsx_vsrli_w(r1, 8);
    r1 = __lsx_vand_v(r1, rb_mask);
    r2 = __lsx_vand_v(src, rb_mask);

    r1 = __lsx_vadd_w(r1, r2);
    t  = __lsx_vand_v(rb_mask, __lsx_vsrli_w(r1, 8));
    r1 = __lsx_vor_v(r1, __lsx_vsub_w(rb_mask_plus_one, t));
    r1 = __lsx_vand_v(r1, rb_mask);

    r2 = __lsx_vsrli_w(dest, 8);
    r2 = __lsx_vand_v(r2, rb_mask);
    r2 = __lsx_vmadd_w(rb_one_half, r2, a);
    t  = __lsx_vand_v(rb_mask, __lsx_vsrli_w(r2, 8));
    r2 = __lsx_vadd_w(r2, t);
    r2 = __lsx_vsrli_w(r2, 8);
    r2 = __lsx_vand_v(r2, rb_mask);
    r3 = __lsx_vand_v(rb_mask, __lsx_vsrli_w(src, 8));

    r2 = __lsx_vadd_w(r2, r3);
    t  = __lsx_vand_v(rb_mask, __lsx_vsrli_w(r2, 8));
    r2 = __lsx_vor_v(r2, __lsx_vsub_w(rb_mask_plus_one, t));
    r2 = __lsx_vand_v(r2, rb_mask);

    t  = __lsx_vor_v(r1, __lsx_vslli_w(r2, 8));

    return t;
}

static force_inline __m128i
lsx_in_u (__m128i x, __m128i a)
{
    __m128i r1, r2, t;
    __m128i rb_mask     = __lsx_vreplgr2vr_w(0xff00ff);
    __m128i rb_one_half = __lsx_vreplgr2vr_w(0x800080);

    r1 = __lsx_vand_v(x, rb_mask);
    r1 = __lsx_vmadd_w(rb_one_half, r1, a);
    t  = __lsx_vand_v(__lsx_vsrli_w(r1, 8), rb_mask);
    r1 = __lsx_vadd_w(r1, t);
    r1 = __lsx_vsrli_w(r1, 8);
    r1 = __lsx_vand_v(r1, rb_mask);
    r2 = __lsx_vsrli_w(x, 8);

    r2 = __lsx_vand_v(r2, rb_mask);
    r2 = __lsx_vmadd_w(rb_one_half, r2, a);
    t  = __lsx_vand_v(__lsx_vsrli_w(r2, 8), rb_mask);
    r2 = __lsx_vadd_w(r2, t);
    r2 = __lsx_vsrli_w(r2, 8);
    r2 = __lsx_vand_v(r2, rb_mask);

    t  = __lsx_vor_v(r1, __lsx_vslli_w(r2, 8));

    return t;
}

static void
lsx_combine_src_u (pixman_implementation_t *imp,
                   pixman_op_t              op,
                   uint32_t *               dest,
                   const uint32_t *         src,
                   const uint32_t *         mask,
                   int                      width)
{
    __m128i src0, mask0, dest0;
    __m128i zero = __lsx_vldi(0);
    __m128i out0, out1, out2, out3;

    if (mask) {
        while (width >= 4) {
            src0 = __lsx_vld(src, 0);
            mask0 = __lsx_vld(mask, 0);
            out0 = __lsx_vilvl_b(zero, src0);
            out2 = __lsx_vilvh_b(zero, src0);
            out1 = __lsx_vilvl_b(zero, mask0);
            out3 = __lsx_vilvh_b(zero, mask0);
            out1 = __lsx_vshuf4i_h(out1, 0xff);
            out3 = __lsx_vshuf4i_h(out3, 0xff);
            out0 = lsx_pix_multiply(out0, out1);
            out2 = lsx_pix_multiply(out2, out3);
            dest0 = __lsx_vpickev_b(out2, out0);
            __lsx_vst(dest0, dest, 0);
            mask  += 4;
            width -= 4;
            src   += 4;
            dest  += 4;
        }

        for (int i = 0; i < width; ++i) {
            uint32_t s = combine_mask(src, mask, i);
            *dest++ = s;
        }
    } else {
        while (width >= 4) {
            src0 = __lsx_vld(src, 0);
            __lsx_vst(src0, dest, 0);
            width -= 4;
            src   += 4;
            dest  += 4;
        }

        if (width) {
            memcpy (dest, src, width * sizeof (uint32_t));
        }
    }
}

static void
lsx_combine_over_u_mask (uint32_t *dest,
                         const uint32_t *src,
                         const uint32_t *mask,
                         int width)
{
    __m128i bit_set = __lsx_vreplgr2vr_h(0xff);
    __m128i src0, mask0, dest0, dest1;
    __m128i zero = __lsx_vldi(0);
    __m128i out0, out1, out2, out3, out4, out5;

    while (width > 3) {
        src0 = __lsx_vld(src, 0);
        dest0 = __lsx_vld(dest, 0);
        mask0 = __lsx_vld(mask, 0);
        out0 = __lsx_vilvl_b(zero, src0);
        out2 = __lsx_vilvh_b(zero, src0);
        out1 = __lsx_vilvl_b(zero, mask0);
        out3 = __lsx_vilvh_b(zero, mask0);
        out1 = __lsx_vshuf4i_h(out1, 0xff);
        out3 = __lsx_vshuf4i_h(out3, 0xff);
        out0 = lsx_pix_multiply(out0, out1);
        out2 = lsx_pix_multiply(out2, out3);
        out1 = __lsx_vxor_v(out0, bit_set);
        out3 = __lsx_vxor_v(out2, bit_set);
        out1 = __lsx_vshuf4i_h(out1, 0xff);
        out3 = __lsx_vshuf4i_h(out3, 0xff);
        out4 = __lsx_vilvl_b(zero, dest0);
        out5 = __lsx_vilvh_b(zero, dest0);
        out4 = lsx_pix_multiply(out4, out1);
        out5 = lsx_pix_multiply(out5, out3);

        dest0 = __lsx_vpickev_b(out2, out0);
        dest1 = __lsx_vpickev_b(out5, out4);
        dest0 = __lsx_vsadd_bu(dest0, dest1);
        __lsx_vst(dest0, dest, 0);
        width -= 4;
        mask  += 4;
        src   += 4;
        dest  += 4;
    }

    for (int i = 0; i < width; ++i) {
        uint32_t m = ALPHA_8 (*(mask + i));
        if (m == 0xFF) {
            uint32_t s = *(src + i);
            uint32_t a = ALPHA_8 (s);
            if (a == 0xFF) {
                *(dest + i) = s;
            } else if (s) {
                uint32_t d = *(dest + i);
                uint32_t ia = a ^ 0xFF;
                UN8x4_MUL_UN8_ADD_UN8x4(d, ia, s);
                *(dest + i) = d;
            }
        } else if (m) {
            uint32_t s = *(src + i);
            if (s) {
                uint32_t d = *(dest + i);
                UN8x4_MUL_UN8(s, m);
                UN8x4_MUL_UN8_ADD_UN8x4(d, ALPHA_8 (~s), s);
                *(dest + i) = d;
            }
        }
    }
}

static void
lsx_combine_over_u_no_mask (uint32_t *dst, const uint32_t *src, int width)
{
    __m128i zero = __lsx_vldi(0);

    while (width >= 4) {
        __m128i v_src, v_dst;
        __m128i v_src_ev, v_src_od;
        __m128i alpha;
        __m128i v_dst_ev, v_dst_od;

        v_src = __lsx_vld(src, 0);
        v_dst = __lsx_vld(dst, 0);

        /* unpack src: 1x128 to 2x128 */
        v_src_ev = __lsx_vpackev_b(zero, v_src);
        v_src_od = __lsx_vpackod_b(zero, v_src);

        /* expand alpha */
        alpha = __lsx_vshuf4i_h(v_src_od, 0xf5);

        /* unpack dst: 1x128 to 2x128 */
        v_dst_ev = __lsx_vpackev_b(zero, v_dst);
        v_dst_od = __lsx_vpackod_b(zero, v_dst);

        v_dst_ev = over_1x128(v_src_ev, alpha, v_dst_ev);
        v_dst_od = over_1x128(v_src_od, alpha, v_dst_od);

        v_dst = __lsx_vpackev_b(v_dst_od, v_dst_ev);

        __lsx_vst(v_dst, dst, 0);
        width -= 4;
        src   += 4;
        dst   += 4;
    }

    while (width--) {
        uint32_t s = *src;
        uint32_t d = *dst;

        *dst = core_combine_over_u32(s, d);

        ++src;
        ++dst;
    }
}

static void
lsx_combine_over_u (pixman_implementation_t *imp,
                    pixman_op_t              op,
                    uint32_t *               dest,
                    const uint32_t *         src,
                    const uint32_t *         mask,
                    int                      width)
{
    if (mask) {
        lsx_combine_over_u_mask(dest, src, mask, width);
    }
    else {
        lsx_combine_over_u_no_mask(dest, src, width);
    }
}

static void
lsx_combine_over_reverse_u (pixman_implementation_t *imp,
                            pixman_op_t              op,
                            uint32_t *               dest,
                            const uint32_t *         src,
                            const uint32_t *         mask,
                            int                      width)
{
    __m128i src0, mask0, dest0, dest1;
    __m128i zero = __lsx_vldi(0);
    __m128i out0, out1, out2, out3, out4, out5;

    if (mask) {
        while (width > 3) {
            src0 = __lsx_vld(src, 0);
            mask0 = __lsx_vld(mask, 0);
            dest0 = __lsx_vld(dest, 0);

            out0 = __lsx_vilvl_b(zero, src0);
            out2 = __lsx_vilvh_b(zero, src0);
            out1 = __lsx_vilvl_b(zero, mask0);
            out3 = __lsx_vilvh_b(zero, mask0);
            out1 = __lsx_vshuf4i_h(out1, 0xff);
            out3 = __lsx_vshuf4i_h(out3, 0xff);
            out0 = lsx_pix_multiply(out0, out1);
            out2 = lsx_pix_multiply(out2, out3);

            dest1 = __lsx_vxori_b(dest0, 0xff);
            out1 = __lsx_vilvl_b(zero, dest0);
            out3 = __lsx_vilvh_b(zero, dest0);
            out4 = __lsx_vilvl_b(zero, dest1);
            out5 = __lsx_vilvh_b(zero, dest1);
            out4 = __lsx_vshuf4i_h(out4, 0xff);
            out5 = __lsx_vshuf4i_h(out5, 0xff);
            out0 = lsx_pix_multiply(out0, out4);
            out2 = lsx_pix_multiply(out2, out5);
            dest0 = __lsx_vpickev_b(out2, out0);
            dest1 = __lsx_vpickev_b(out3, out1);
            dest0 = __lsx_vsadd_bu(dest0, dest1);
            __lsx_vst(dest0, dest, 0);
            mask  += 4;
            width -= 4;
            src   += 4;
            dest  += 4;
        }
    } else {
        while (width > 3) {
            src0  = __lsx_vld(src, 0);
            dest0 = __lsx_vld(dest, 0);
            dest1 = __lsx_vxori_b(dest0, 0xff);
            out0 = __lsx_vilvl_b(zero, src0);
            out2 = __lsx_vilvh_b(zero, src0);
            out1 = __lsx_vilvl_b(zero, dest0);
            out3 = __lsx_vilvh_b(zero, dest0);
            out4 = __lsx_vilvl_b(zero, dest1);
            out5 = __lsx_vilvh_b(zero, dest1);
            out4 = __lsx_vshuf4i_h(out4, 0xff);
            out5 = __lsx_vshuf4i_h(out5, 0xff);
            out0 = lsx_pix_multiply(out0, out4);
            out2 = lsx_pix_multiply(out2, out5);
            dest0 = __lsx_vpickev_b(out2, out0);
            dest1 = __lsx_vpickev_b(out3, out1);
            dest0 = __lsx_vsadd_bu(dest0, dest1);
            __lsx_vst(dest0, dest, 0);
            width -= 4;
            src   += 4;
            dest  += 4;
        }
    }

    for (int i = 0; i < width; ++i) {
        uint32_t s = combine_mask(src, mask, i);
        uint32_t d = *(dest + i);
        uint32_t ia = ALPHA_8 (~*(dest + i));
        UN8x4_MUL_UN8_ADD_UN8x4(s, ia, d);
        *(dest + i) = s;
    }
}

static void
lsx_combine_out_u (pixman_implementation_t *imp,
                   pixman_op_t              op,
                   uint32_t *               dest,
                   const uint32_t *         src,
                   const uint32_t *         mask,
                   int                      width)
{
    __m128i src0, mask0, dest0;
    __m128i zero = __lsx_vldi(0);
    __m128i out0, out1, out2, out3;

    if (mask) {
        while (width > 3) {
            src0 = __lsx_vld(src, 0);
            mask0 = __lsx_vld(mask, 0);
            dest0 = __lsx_vld(dest, 0);

            out0 = __lsx_vilvl_b(zero, src0);
            out2 = __lsx_vilvh_b(zero, src0);
            out1 = __lsx_vilvl_b(zero, mask0);
            out3 = __lsx_vilvh_b(zero, mask0);
            out1 = __lsx_vshuf4i_h(out1, 0xff);
            out3 = __lsx_vshuf4i_h(out3, 0xff);
            out0 = lsx_pix_multiply(out0, out1);
            out2 = lsx_pix_multiply(out2, out3);

            dest0 = __lsx_vxori_b(dest0, 0xff);
            out1 = __lsx_vilvl_b(zero, dest0);
            out3 = __lsx_vilvh_b(zero, dest0);
            out1 = __lsx_vshuf4i_h(out1, 0xff);
            out3 = __lsx_vshuf4i_h(out3, 0xff);
            out0 = lsx_pix_multiply(out0, out1);
            out2 = lsx_pix_multiply(out2, out3);
            dest0 = __lsx_vpickev_b(out2, out0);
            __lsx_vst(dest0, dest, 0);
            mask  += 4;
            width -= 4;
            src   += 4;
            dest  += 4;
        }
    } else {
        while (width > 3) {
            src0 = __lsx_vld(src, 0);
            dest0 = __lsx_vld(dest, 0);
            out0 = __lsx_vilvl_b(zero, src0);
            out2 = __lsx_vilvh_b(zero, src0);
            dest0 = __lsx_vxori_b(dest0, 0xff);
            out1 = __lsx_vilvl_b(zero, dest0);
            out3 = __lsx_vilvh_b(zero, dest0);
            out1 = __lsx_vshuf4i_h(out1, 0xff);
            out3 = __lsx_vshuf4i_h(out3, 0xff);
            out0 = lsx_pix_multiply(out0, out1);
            out2 = lsx_pix_multiply(out2, out3);
            dest0 = __lsx_vpickev_b(out2, out0);
            __lsx_vst(dest0, dest, 0);
            width -= 4;
            src   += 4;
            dest  += 4;
        }
    }

    for (int i = 0; i < width; ++i) {
        uint32_t s = combine_mask(src, mask, i);
        uint32_t a = ALPHA_8 (~*(dest + i));
        UN8x4_MUL_UN8(s, a);
        *(dest + i) = s;
    }
}

static void
lsx_combine_out_reverse_u (pixman_implementation_t *imp,
                           pixman_op_t              op,
                           uint32_t *               dest,
                           const uint32_t *         src,
                           const uint32_t *         mask,
                           int                      width)
{
    __m128i bit_set = __lsx_vreplgr2vr_h(0xff);
    __m128i src0, mask0, dest0;
    __m128i zero = __lsx_vldi(0);
    __m128i out0, out1, out2, out3;

    if (mask) {
        while (width > 3) {
            src0 = __lsx_vld(src, 0);
            dest0 = __lsx_vld(dest, 0);
            mask0 = __lsx_vld(mask, 0);

            out0 = __lsx_vilvl_b(zero, src0);
            out2 = __lsx_vilvh_b(zero, src0);
            out1 = __lsx_vilvl_b(zero, mask0);
            out3 = __lsx_vilvh_b(zero, mask0);
            out1 = __lsx_vshuf4i_h(out1, 0xff);
            out3 = __lsx_vshuf4i_h(out3, 0xff);
            out0 = lsx_pix_multiply(out0, out1);
            out2 = lsx_pix_multiply(out2, out3);
            out1 = __lsx_vxor_v(out0, bit_set);
            out3 = __lsx_vxor_v(out2, bit_set);
            out1 = __lsx_vshuf4i_h(out1, 0xff);
            out3 = __lsx_vshuf4i_h(out3, 0xff);
            out0 = __lsx_vilvl_b(zero, dest0);
            out2 = __lsx_vilvh_b(zero, dest0);
            out0 = lsx_pix_multiply(out0, out1);
            out2 = lsx_pix_multiply(out2, out3);
            dest0 = __lsx_vpickev_b(out2, out0);
            __lsx_vst(dest0, dest, 0);
            mask  += 4;
            width -= 4;
            src   += 4;
            dest  += 4;
        }
    } else {
        while (width > 3) {
            src0 = __lsx_vld(src, 0);
            dest0 = __lsx_vld(dest, 0);
            out0 = __lsx_vilvl_b(zero, src0);
            out2 = __lsx_vilvh_b(zero, src0);
            out1 = __lsx_vxor_v(out0, bit_set);
            out3 = __lsx_vxor_v(out2, bit_set);
            out1 = __lsx_vshuf4i_h(out1, 0xff);
            out3 = __lsx_vshuf4i_h(out3, 0xff);
            out0 = __lsx_vilvl_b(zero, dest0);
            out2 = __lsx_vilvh_b(zero, dest0);
            out0 = lsx_pix_multiply(out0, out1);
            out2 = lsx_pix_multiply(out2, out3);
            dest0 = __lsx_vpickev_b(out2, out0);
            __lsx_vst(dest0, dest, 0);
            width -= 4;
            src   += 4;
            dest  += 4;
        }
    }

    for (int i = 0; i < width; ++i) {
        uint32_t s = combine_mask(src, mask, i);
        uint32_t d = *(dest + i);
        uint32_t a = ALPHA_8 (~s);
        UN8x4_MUL_UN8 (d, a);
        *(dest + i) = d;
    }
}

static void
lsx_combine_add_u (pixman_implementation_t *imp,
                   pixman_op_t              op,
                   uint32_t *               dest,
                   const uint32_t *         src,
                   const uint32_t *         mask,
                   int                      width)
{
    __m128i src0, mask0, dest0, dest1;
    __m128i zero = __lsx_vldi(0);
    __m128i out0, out1, out2, out3;

    if (mask) {
        while (width > 3) {
            src0 = __lsx_vld(src, 0);
            dest0 = __lsx_vld(dest, 0);
            mask0 = __lsx_vld(mask, 0);

            out0 = __lsx_vilvl_b(zero, src0);
            out2 = __lsx_vilvh_b(zero, src0);
            out1 = __lsx_vilvl_b(zero, mask0);
            out3 = __lsx_vilvh_b(zero, mask0);
            out1 = __lsx_vshuf4i_h(out1, 0xff);
            out3 = __lsx_vshuf4i_h(out3, 0xff);
            out0 = lsx_pix_multiply(out0, out1);
            out2 = lsx_pix_multiply(out2, out3);

            dest1 = __lsx_vpickev_b(out2, out0);
            dest0 = __lsx_vsadd_bu(dest0, dest1);
            __lsx_vst(dest0, dest, 0);
            mask  += 4;
            width -= 4;
            src   += 4;
            dest  += 4;
        }
    } else {
        while (width > 3) {
            src0 = __lsx_vld(src, 0);
            dest0 = __lsx_vld(dest, 0);
            out0 = __lsx_vilvl_b(zero, src0);
            out2 = __lsx_vilvh_b(zero, src0);
            dest1 = __lsx_vpickev_b(out2, out0);
            dest0 = __lsx_vsadd_bu(dest0, dest1);
            __lsx_vst(dest0, dest, 0);
            width -= 4;
            src   += 4;
            dest  += 4;
        }
    }

    for (int i = 0; i < width; ++i) {
        uint32_t s = combine_mask(src, mask, i);
        uint32_t d = *(dest + i);
        UN8x4_ADD_UN8x4(d, s);
        *(dest + i) = d;
    }
}

/*
 * Multiply
 *
 *      ad * as * B(d / ad, s / as)
 *    = ad * as * d/ad * s/as
 *    = d * s
 *
 */
static void
lsx_combine_multiply_u (pixman_implementation_t *imp,
                        pixman_op_t              op,
                        uint32_t *               dest,
                        const uint32_t *         src,
                        const uint32_t *         mask,
                        int                      width)
{
    __m128i bit_set = __lsx_vreplgr2vr_h(0xff);
    __m128i src0, mask0, dest0, dest1;
    __m128i zero = __lsx_vldi(0);
    __m128i out0, out1, out2, out3, out4, out5, out6, out7;

    if (mask) {
        while (width > 3) {
            src0 = __lsx_vld(src, 0);
            dest0 = __lsx_vld(dest, 0);
            mask0 = __lsx_vld(mask, 0);

            out0 = __lsx_vilvl_b(zero, src0);
            out2 = __lsx_vilvh_b(zero, src0);
            out1 = __lsx_vilvl_b(zero, mask0);
            out3 = __lsx_vilvh_b(zero, mask0);
            out1 = __lsx_vshuf4i_h(out1, 0xff);
            out3 = __lsx_vshuf4i_h(out3, 0xff);
            out0 = lsx_pix_multiply(out0, out1);
            out2 = lsx_pix_multiply(out2, out3);

            out1 = __lsx_vxor_v(out0, bit_set);
            out3 = __lsx_vxor_v(out2, bit_set);
            out1 = __lsx_vshuf4i_h(out1, 0xff);
            out3 = __lsx_vshuf4i_h(out3, 0xff);
            dest1 = __lsx_vxori_b(dest0, 0xff);
            dest1 = __lsx_vshuf4i_b(dest1, 0xff);
            out4 = __lsx_vilvl_b(zero, dest0);
            out5 = __lsx_vilvh_b(zero, dest0);
            out6 = __lsx_vilvl_b(zero, dest1);
            out7 = __lsx_vilvh_b(zero, dest1);
            out6 = lsx_pix_multiply(out0, out6);
            out7 = lsx_pix_multiply(out2, out7);
            out1 = lsx_pix_multiply(out4, out1);
            out3 = lsx_pix_multiply(out5, out3);
            dest0 = __lsx_vpickev_b(out7, out6);
            dest1 = __lsx_vpickev_b(out3, out1);
            dest0 = __lsx_vsadd_bu(dest0, dest1);

            out4 = lsx_pix_multiply(out4, out0);
            out5 = lsx_pix_multiply(out5, out2);
            dest1 = __lsx_vpickev_b(out5, out4);
            dest0 = __lsx_vsadd_bu(dest0, dest1);
            __lsx_vst(dest0, dest, 0);
            mask  += 4;
            width -= 4;
            src   += 4;
            dest  += 4;
        }
    } else {
        while (width > 3) {
            src0 = __lsx_vld(src, 0);
            dest0 = __lsx_vld(dest, 0);
            out0 = __lsx_vilvl_b(zero, src0);
            out2 = __lsx_vilvh_b(zero, src0);
            out1 = __lsx_vxor_v(out0, bit_set);
            out3 = __lsx_vxor_v(out2, bit_set);
            out1 = __lsx_vshuf4i_h(out1, 0xff);
            out3 = __lsx_vshuf4i_h(out3, 0xff);
            dest1 = __lsx_vxori_b(dest0, 0xff);
            dest1 = __lsx_vshuf4i_b(dest1, 0xff);
            out4 = __lsx_vilvl_b(zero, dest0);
            out5 = __lsx_vilvh_b(zero, dest0);
            out6 = __lsx_vilvl_b(zero, dest1);
            out7 = __lsx_vilvh_b(zero, dest1);
            out6 = lsx_pix_multiply(out0, out6);
            out7 = lsx_pix_multiply(out2, out7);
            out1 = lsx_pix_multiply(out4, out1);
            out3 = lsx_pix_multiply(out5, out3);
            dest0 = __lsx_vpickev_b(out7, out6);
            dest1 = __lsx_vpickev_b(out3, out1);
            dest0 = __lsx_vsadd_bu(dest0, dest1);

            out4 = lsx_pix_multiply(out4, out0);
            out5 = lsx_pix_multiply(out5, out2);
            dest1 = __lsx_vpickev_b(out5, out4);
            dest0 = __lsx_vsadd_bu(dest0, dest1);
            __lsx_vst(dest0, dest, 0);
            width -= 4;
            src   += 4;
            dest  += 4;
        }
    }

    for (int i = 0; i < width; ++i) {
        uint32_t s = combine_mask(src, mask, i);
        uint32_t d = *(dest + i);
        uint32_t ss = s;
        uint32_t src_ia = ALPHA_8(~s);
        uint32_t dest_ia = ALPHA_8(~d);

        UN8x4_MUL_UN8_ADD_UN8x4_MUL_UN8(ss, dest_ia, d, src_ia);
        UN8x4_MUL_UN8x4(d, s);
        UN8x4_ADD_UN8x4(d, ss);

        *(dest + i) = d;
    }
}

static void
lsx_combine_src_ca (pixman_implementation_t *imp,
                    pixman_op_t              op,
                    uint32_t *               dest,
                    const uint32_t *         src,
                    const uint32_t *         mask,
                    int                      width)
{
    __m128i src0, mask0, dest0;
    __m128i zero = __lsx_vldi(0);
    __m128i out0, out1, out2, out3;

    while (width > 3) {
        src0 = __lsx_vld(src, 0);
        mask0 = __lsx_vld(mask, 0);
        out0 = __lsx_vilvl_b(zero, src0);
        out2 = __lsx_vilvh_b(zero, src0);
        out1 = __lsx_vilvl_b(zero, mask0);
        out3 = __lsx_vilvh_b(zero, mask0);
        out0 = lsx_pix_multiply(out0, out1);
        out2 = lsx_pix_multiply(out2, out3);
        dest0 = __lsx_vpickev_b(out2, out0);
        __lsx_vst(dest0, dest, 0);
        mask  += 4;
        width -= 4;
        src   += 4;
        dest  += 4;
    }

    for (int i = 0; i < width; ++i) {
        uint32_t s = *(src + i);
        uint32_t m = *(mask + i);
        combine_mask_value_ca(&s, &m);
        *(dest + i) = s;
    }
}

static void
lsx_combine_over_ca (pixman_implementation_t  *imp,
                     pixman_op_t               op,
                     uint32_t *                dest,
                     const uint32_t *          src,
                     const uint32_t *          mask,
                     int                       width)
{
    __m128i bit_set = __lsx_vreplgr2vr_h(0xff);
    __m128i src0, mask0, dest0, dest1;
    __m128i zero = __lsx_vldi(0);
    __m128i out0, out1, out2, out3, out4, out5;

    while (width > 3) {
        src0 = __lsx_vld(src, 0);
        dest0 = __lsx_vld(dest, 0);
        mask0 = __lsx_vld(mask, 0);

        out0 = __lsx_vilvl_b(zero, src0);
        out2 = __lsx_vilvh_b(zero, src0);
        out1 = __lsx_vilvl_b(zero, mask0);
        out3 = __lsx_vilvh_b(zero, mask0);
        out4 = lsx_pix_multiply(out0, out1);
        out5 = lsx_pix_multiply(out2, out3);
        out0 = __lsx_vshuf4i_h(out0, 0xff);
        out2 = __lsx_vshuf4i_h(out2, 0xff);
        out1 = lsx_pix_multiply(out1, out0);
        out3 = lsx_pix_multiply(out3, out2);

        out1 = __lsx_vxor_v(out1, bit_set);
        out3 = __lsx_vxor_v(out3, bit_set);
        out0 = __lsx_vilvl_b(zero, dest0);
        out2 = __lsx_vilvh_b(zero, dest0);
        out1 = lsx_pix_multiply(out1, out0);
        out3 = lsx_pix_multiply(out3, out2);

        dest0 = __lsx_vpickev_b(out5, out4);
        dest1 = __lsx_vpickev_b(out3, out1);
        dest0 = __lsx_vsadd_bu(dest0, dest1);
        __lsx_vst(dest0, dest, 0);
        mask  += 4;
        width -= 4;
        src   += 4;
        dest  += 4;
    }

    for (int i = 0; i < width; ++i) {
        uint32_t s = *(src + i);
        uint32_t m = *(mask + i);
        uint32_t a;

        combine_mask_ca (&s, &m);
        a = ~m;
        if (a) {
            uint32_t d = *(dest + i);
            UN8x4_MUL_UN8x4_ADD_UN8x4(d, a, s);
            s = d;
        }
        *(dest + i) = s;
    }
}

static void
lsx_combine_out_reverse_ca (pixman_implementation_t *imp,
                            pixman_op_t              op,
                            uint32_t *               dest,
                            const uint32_t *         src,
                            const uint32_t *         mask,
                            int                      width)
{
    __m128i bit_set = __lsx_vreplgr2vr_h(0xff);
    __m128i src0, mask0, dest0;
    __m128i zero = __lsx_vldi(0);
    __m128i out0, out1, out2, out3;

    while (width > 3) {
        src0 = __lsx_vld(src, 0);
        dest0 = __lsx_vld(dest, 0);
        mask0 = __lsx_vld(mask, 0);

        out0 = __lsx_vilvl_b(zero, src0);
        out2 = __lsx_vilvh_b(zero, src0);
        out1 = __lsx_vilvl_b(zero, mask0);
        out3 = __lsx_vilvh_b(zero, mask0);
        out0 = __lsx_vshuf4i_h(out0, 0xff);
        out2 = __lsx_vshuf4i_h(out2, 0xff);
        out1 = lsx_pix_multiply(out1, out0);
        out3 = lsx_pix_multiply(out3, out2);

        out1 = __lsx_vxor_v(out1, bit_set);
        out3 = __lsx_vxor_v(out3, bit_set);
        out0 = __lsx_vilvl_b(zero, dest0);
        out2 = __lsx_vilvh_b(zero, dest0);
        out1 = lsx_pix_multiply(out1, out0);
        out3 = lsx_pix_multiply(out3, out2);
        dest0 = __lsx_vpickev_b(out3, out1);
        __lsx_vst(dest0, dest, 0);
        mask  += 4;
        width -= 4;
        src   += 4;
        dest  += 4;
    }

    for (int i = 0; i < width; ++i) {
        uint32_t s = *(src + i);
        uint32_t m = *(mask + i);
        uint32_t a;

        combine_mask_alpha_ca(&s, &m);
        a = ~m;

        if (a != ~0) {
            uint32_t d = 0;

            if (a) {
                d = *(dest + i);
                UN8x4_MUL_UN8x4(d, a);
            }
            *(dest + i) = d;
        }
    }
}

/*
 * w : length in bytes
 */
static void force_inline
lsx_blt_one_line_u8 (uint8_t *pDst, uint8_t *pSrc, int w)
{
    /* align the dst to 16 byte */
    while (((uintptr_t)pDst & 15) && w) {
        *pDst = *pSrc;
        pSrc += 1;
        pDst += 1;
        w -= 1;
    }

    while (w >= 32) {
        __m128i src0, src1;
        src0 = __lsx_vld(pSrc, 0);
        src1 = __lsx_vld(pSrc, 16);
        __lsx_vst(src0, pDst, 0);
        __lsx_vst(src1, pDst, 16);

        w -= 32;
        pSrc += 32;
        pDst += 32;
    }

    if (w >= 16) {
        __lsx_vst(__lsx_vld(pSrc, 0), pDst, 0);

        w -= 16;
        pSrc += 16;
        pDst += 16;
    }

    if (w >= 8) {
        *(uint64_t *)pDst = *(uint64_t *)pSrc;

        w -= 8;
        pSrc += 8;
        pDst += 8;
    }

    while (w--) {
        /* copy one bytes once a time */
        *pDst++ = *pSrc++;
    }
}

/*
 * w : length in half word
 */
static void
lsx_blt_one_line_u16 (uint16_t *pDst, uint16_t *pSrc, int w)
{
    /* align the dst to 16 byte */
    while (((uintptr_t)pDst & 15) && w) {
        *pDst++ = *pSrc++;
        --w;
    }

    while (w >= 32) {
        __m128i src0, src1, src2, src3;
        /* copy 64 bytes */
        src0 = __lsx_vld(pSrc, 0);
        src1 = __lsx_vld(pSrc, 16);
        src2 = __lsx_vld(pSrc, 32);
        src3 = __lsx_vld(pSrc, 48);
        __lsx_vst(src0, pDst, 0);
        __lsx_vst(src1, pDst, 16);
        __lsx_vst(src2, pDst, 32);
        __lsx_vst(src3, pDst, 48);

        w -= 32;
        pSrc += 32;
        pDst += 32;
    }

    if (w >= 16) {
        __m128i src0, src1;
        /* copy 32 bytes */
        src0 = __lsx_vld(pSrc, 0);
        src1 = __lsx_vld(pSrc, 16);
        __lsx_vst(src0, pDst, 0);
        __lsx_vst(src1, pDst, 16);

        w -= 16;
        pSrc += 16;
        pDst += 16;
    }

    if (w >= 8) {
        /* copy 16 bytes */
        __lsx_vst(__lsx_vld(pSrc, 0), pDst, 0);

        w -= 8;
        pSrc += 8;
        pDst += 8;
    }

    while (w--) {
        /* copy 2 bytes once a time */
        *pDst++ = *pSrc++;
    }
}

/*
 * w : length in word
 */
static force_inline void
lsx_blt_one_line_u32 (uint32_t *pDst, uint32_t *pSrc, int w)
{
    /* align the dst to 16 byte */
    while (((uintptr_t)pDst & 15) && w) {
        *pDst++ = *pSrc++;
        --w;
    }

    while (w >= 32) {
        __m128i src0, src1, src2, src3;
        __m128i src4, src5, src6, src7;
        /* copy 128 bytes */
        src0 = __lsx_vld(pSrc, 0);
        src1 = __lsx_vld(pSrc, 16);
        src2 = __lsx_vld(pSrc, 32);
        src3 = __lsx_vld(pSrc, 48);
        src4 = __lsx_vld(pSrc, 64);
        src5 = __lsx_vld(pSrc, 80);
        src6 = __lsx_vld(pSrc, 96);
        src7 = __lsx_vld(pSrc, 112);
        __lsx_vst(src0, pDst, 0);
        __lsx_vst(src1, pDst, 16);
        __lsx_vst(src2, pDst, 32);
        __lsx_vst(src3, pDst, 48);
        __lsx_vst(src4, pDst, 64);
        __lsx_vst(src5, pDst, 80);
        __lsx_vst(src6, pDst, 96);
        __lsx_vst(src7, pDst, 112);

        w -= 32;
        pSrc += 32;
        pDst += 32;
    }

    if (w >= 16) {
        __m128i src0, src1, src2, src3;
        /* copy 64 bytes */
        src0 = __lsx_vld(pSrc, 0);
        src1 = __lsx_vld(pSrc, 16);
        src2 = __lsx_vld(pSrc, 32);
        src3 = __lsx_vld(pSrc, 48);
        __lsx_vst(src0, pDst, 0);
        __lsx_vst(src1, pDst, 16);
        __lsx_vst(src2, pDst, 32);
        __lsx_vst(src3, pDst, 48);

        w -= 16;
        pSrc += 16;
        pDst += 16;
    }

    if (w >= 8) {
        __m128i src0, src1;
        /* copy 32 bytes */
        src0 = __lsx_vld(pSrc, 0);
        src1 = __lsx_vld(pSrc, 16);
        __lsx_vst(src0, pDst, 0);
        __lsx_vst(src1, pDst, 16);

        w -= 8;
        pSrc += 8;
        pDst += 8;
    }

    if (w >= 4) {
        /* copy 16 bytes once a time */
        __lsx_vst(__lsx_vld(pSrc, 0), pDst, 0);

        w -= 4;
        pSrc += 4;
        pDst += 4;
    }

    while (w--) {
        /* copy 4 bytes once a time */
        *pDst++ = *pSrc++;
    }
}

static pixman_bool_t
lsx_blt (pixman_implementation_t *imp,
         uint32_t *               src_bits,
         uint32_t *               dst_bits,
         int                      src_stride,
         int                      dst_stride,
         int                      src_bpp,
         int                      dst_bpp,
         int                      src_x,
         int                      src_y,
         int                      dest_x,
         int                      dest_y,
         int                      width,
         int                      height)
{
    if (src_bpp != dst_bpp)
        return FALSE;

    if (src_bpp == 8) {
        uint8_t *src_b = (uint8_t *)src_bits;
        uint8_t *dst_b = (uint8_t *)dst_bits;

        src_stride = src_stride * 4;
        dst_stride = dst_stride * 4;

        src_b += src_stride * src_y + src_x;
        dst_b += dst_stride * dest_y + dest_x;

        while (height--) {
            lsx_blt_one_line_u8 (dst_b, src_b, width);
            dst_b += dst_stride;
            src_b += src_stride;
        }

        return TRUE;
    }

    if (src_bpp == 16) {
        uint16_t *src_h = (uint16_t *)src_bits;
        uint16_t *dst_h = (uint16_t *)dst_bits;

        src_stride = src_stride * 2;
        dst_stride = dst_stride * 2;

        src_h += src_stride * src_y + src_x;
        dst_h += dst_stride * dest_y + dest_x;

        while (height--) {
            lsx_blt_one_line_u16 (dst_h, src_h, width);
            dst_h += dst_stride;
            src_h += src_stride;
        }

        return TRUE;
    }

    if (src_bpp == 32) {
        src_bits += src_stride * src_y + src_x;
        dst_bits += dst_stride * dest_y + dest_x;

        while (height--) {
            lsx_blt_one_line_u32 (dst_bits, src_bits, width);
            dst_bits += dst_stride;
            src_bits += src_stride;
        }

        return TRUE;
    }

    return FALSE;
}

static void
lsx_fill_u8 (uint8_t  *dst,
             int       stride,
             int       x,
             int       y,
             int       width,
             int       height,
             uint8_t   filler)
{
    __m128i vfill = __lsx_vreplgr2vr_b(filler);
    int byte_stride = stride * 4;
    dst += y * byte_stride + x;

    while (height--) {
        int w = width;
        uint8_t *d = dst;

        while (w && ((uintptr_t)d & 15)) {
            *d = filler;
            w--;
            d++;
        }

        while (w >= 64) {
            __lsx_vst(vfill, d, 0);
            __lsx_vst(vfill, d, 16);
            __lsx_vst(vfill, d, 32);
            __lsx_vst(vfill, d, 48);
            w -= 64;
            d += 64;
        }

        if (w >= 32) {
            __lsx_vst(vfill, d, 0);
            __lsx_vst(vfill, d, 16);
            w -= 32;
            d += 32;
        }

        if (w >= 16) {
            __lsx_vst(vfill, d, 0);
            w -= 16;
            d += 16;
        }

        while (w) {
            *d = filler;
            w--;
            d++;
        }

        dst += byte_stride;
    }
}

static void
lsx_fill_u16 (uint16_t *dst,
              int       stride,
              int       x,
              int       y,
              int       width,
              int       height,
              uint16_t  filler)
{
    __m128i vfill = __lsx_vreplgr2vr_h(filler);
    int short_stride = stride * 2;
    dst += y * short_stride + x;

    while (height--) {
        int w = width;
        uint16_t *d = dst;

        while (w && ((uintptr_t)d & 15)) {
            *d = filler;
            w--;
            d++;
        }

        while (w >= 32) {
            __lsx_vst(vfill, d, 0);
            __lsx_vst(vfill, d, 16);
            __lsx_vst(vfill, d, 32);
            __lsx_vst(vfill, d, 48);
            w -= 32;
            d += 32;
        }

        if (w >= 16) {
            __lsx_vst(vfill, d, 0);
            __lsx_vst(vfill, d, 16);
            w -= 16;
            d += 16;
        }

        if (w >= 8) {
            __lsx_vst(vfill, d, 0);
            w -= 8;
            d += 8;
        }

        while (w) {
            *d = filler;
            w--;
            d++;
        }

        dst += short_stride;
    }
}

static void
lsx_fill_u32 (uint32_t *bits,
              int       stride,
              int       x,
              int       y,
              int       width,
              int       height,
              uint32_t  filler)
{
    __m128i vfill = __lsx_vreplgr2vr_w(filler);
    bits += y * stride + x;

    while (height--) {
        int w = width;
        uint32_t *d = bits;

        while (w && ((uintptr_t)d & 15)) {
            *d = filler;
            w--;
            d++;
        }

        while (w >= 32) {
            __lsx_vst(vfill, d, 0);
            __lsx_vst(vfill, d, 16);
            __lsx_vst(vfill, d, 32);
            __lsx_vst(vfill, d, 48);
            __lsx_vst(vfill, d, 64);
            __lsx_vst(vfill, d, 80);
            __lsx_vst(vfill, d, 96);
            __lsx_vst(vfill, d, 112);
            w -= 32;
            d += 32;
        }

        while (w >= 16) {
            __lsx_vst(vfill, d, 0);
            __lsx_vst(vfill, d, 16);
            __lsx_vst(vfill, d, 32);
            __lsx_vst(vfill, d, 48);
            w -= 16;
            d += 16;
        }

        if (w >= 8) {
            __lsx_vst(vfill, d, 0);
            __lsx_vst(vfill, d, 16);
            w -= 8;
            d += 8;
        }

        if (w >= 4) {
            __lsx_vst(vfill, d, 0);
            w -= 4;
            d += 4;
        }

        while (w) {
            *d = filler;
            w--;
            d++;
        }

        bits += stride;
    }
}

static pixman_bool_t
lsx_fill (pixman_implementation_t *imp,
          uint32_t *               bits,
          int                      stride,
          int                      bpp,
          int                      x,
          int                      y,
          int                      width,
          int                      height,
          uint32_t                 filler)
{
    switch (bpp) {
        case 8:
            lsx_fill_u8 ((uint8_t *)bits, stride, x, y, width, height, (uint8_t)filler);
            return TRUE;

        case 16:
            lsx_fill_u16 ((uint16_t *)bits, stride, x, y, width, height, (uint16_t)filler);
            return TRUE;

        case 32:
            lsx_fill_u32 (bits, stride, x, y, width, height, filler);
            return TRUE;

        default:
            return FALSE;
    }

    return TRUE;
}

static void
lsx_composite_over_n_8_8888 (pixman_implementation_t *imp,
                             pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t src, srca;
    uint32_t *dst_line, *dst, d;
    uint8_t  *mask_line, *mask, m;
    int dst_stride, mask_stride;
    int32_t w;
    v4u32 vsrca, vsrc;
    __m128i vff;

    src   = _pixman_image_get_solid(imp, src_image, dest_image->bits.format);
    vsrc  = (v4u32)__lsx_vreplgr2vr_w(src);
    srca  = src >> 24;
    vsrca = (v4u32)__lsx_vreplgr2vr_w(srca);
    vff   = __lsx_vreplgr2vr_w(0xff);

    if (src == 0)
        return;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (mask_image, mask_x, mask_y, uint8_t, mask_stride, mask_line, 1);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        mask = mask_line;
        mask_line += mask_stride;
        w = width;

        while (w >= 4) {
            v4u32 ma = {mask[0], mask[1], mask[2], mask[3]};

            if (__lsx_bnz_w(__lsx_vseqi_w((__m128i)ma, 0xff))){
                if (__lsx_bnz_w(__lsx_vseqi_w(vsrca, 0xff)))
                    *(__m128i*) dst = (__m128i)vsrc;
                else if (__lsx_bnz_w(__lsx_vsub_w((__m128i)ma, vff)))
                    *(__m128i*) dst = lsx_over_u((__m128i)vsrc, *(__m128i*)dst);
            } else if (__lsx_bnz_w((__m128i)ma)) {
                __m128i d0 = lsx_in_u((__m128i)vsrc, (__m128i)ma);
                *(__m128i*) dst = lsx_over_u(d0, *(__m128i*)dst);
            } else {
                for (int i = 0; i < 4; i++) {
                    if (mask[i] == 0xff) {
                        if (vsrca[i] == 0xff)
                            *(dst + i) = vsrc[i];
                        else
                            *(dst + i) = over(vsrc[i], *(dst + i));
                    } else if (mask[i]) {
                        m = mask[i];
                        d = in(vsrc[i], m);
                        *(dst + i) = over(d, *(dst + i));
                    }
                }
            }
            dst += 4;
            w -= 4;
            mask += 4;
        }

        while (w--) {
            m = *mask++;
            if (m == 0xff) {
                if (srca == 0xff)
                    *dst = src;
                else
                    *dst = over(src, *dst);
            } else if (m) {
                d = in(src, m);
                *dst = over(d, *dst);
            }
            dst++;
        }
    }
}

static void
lsx_composite_add_8_8 (pixman_implementation_t *imp,
                       pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint8_t *dst_line, *dst;
    uint8_t *src_line, *src;
    int dst_stride, src_stride;
    int32_t w;
    uint16_t t;

    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint8_t, src_stride, src_line, 1);
    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint8_t, dst_stride, dst_line, 1);

    while (height--) {
        dst = dst_line;
        src = src_line;

        dst_line += dst_stride;
        src_line += src_stride;
        w = width;

        lsx_combine_add_u(imp, op, (uint32_t *)dst, (uint32_t *)src, NULL, w >> 2);
        dst += w & 0xfffc;
        src += w & 0xfffc;
        w &= 3;

        while (w--) {
            t = (*dst) + (*src++);
            *dst++ = t | (0 - (t >> 8));
        }
    }
}

static void
lsx_composite_add_8888_8888 (pixman_implementation_t *imp,
                             pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t *dst_line;
    uint32_t *src_line;
    int dst_stride, src_stride;

    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint32_t, src_stride, src_line, 1);
    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);

    while (height--) {
        lsx_combine_add_u(imp, op, dst_line, src_line, NULL, width);
        dst_line += dst_stride;
        src_line += src_stride;
    }
}

static void
lsx_composite_over_8888_8888 (pixman_implementation_t *imp,
                              pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    int dst_stride, src_stride;
    uint32_t *dst_line;
    uint32_t *src_line;

    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint32_t, src_stride, src_line, 1);
    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);

    while (height--) {
        lsx_combine_over_u_no_mask (dst_line, src_line, width);
        dst_line += dst_stride;
        src_line += src_stride;
    }
}

static void
lsx_composite_copy_area (pixman_implementation_t *imp,
                         pixman_composite_info_t *info)
{
    bits_image_t src_bits, dst_bits;
    src_bits = info->src_image->bits;
    dst_bits = info->dest_image->bits;
    lsx_blt (imp, src_bits.bits,
             dst_bits.bits,
             src_bits.rowstride,
             dst_bits.rowstride,
             PIXMAN_FORMAT_BPP (src_bits.format),
             PIXMAN_FORMAT_BPP (dst_bits.format),
             info->src_x, info->src_y, info->dest_x,
             info->dest_y, info->width, info->height);
}

static void
lsx_composite_src_x888_0565 (pixman_implementation_t *imp,
                             pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint16_t *dst_line, *dst;
    uint32_t *src_line, *src, s;
    int dst_stride, src_stride;
    int32_t w;

    __m128i src0, tmp;
    __m128i rb0, t0, g0;
    __m128i mask_565_rb = __lsx_vreplgr2vr_w(0x001f001f);
    __m128i mask_green_4x32 = __lsx_vreplgr2vr_w(0x0000fc00);

    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint32_t, src_stride, src_line, 1);
    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint16_t, dst_stride, dst_line, 1);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        src = src_line;
        src_line += src_stride;
        w = width;

        while (w && (uintptr_t)src & 15) {
            s = *src++;
            *dst = convert_8888_to_0565(s);
            dst++;
            w--;
        }

        while (w >= 4) {
            src0 = __lsx_vld(src, 0);
            src += 4;
            w -= 4;

            rb0 = __lsx_vsrli_w(src0, 3) & mask_565_rb;
            g0 = src0 & mask_green_4x32;
            rb0 = rb0 | __lsx_vsrli_w(rb0, 5);
            t0 = rb0 | __lsx_vsrli_w(g0, 5);
            tmp = __lsx_vpickev_h(t0, t0);
            __lsx_vstelm_d(tmp, dst, 0, 0);
            dst += 4;
        }

        while (w--) {
            s = *src++;
            *dst = convert_8888_to_0565(s);
            dst++;
        }
    }
}

static void
lsx_composite_in_n_8_8 (pixman_implementation_t *imp,
                        pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS(info);
    uint8_t *dst_line, *dst;
    uint8_t *mask_line, *mask;
    int dst_stride, mask_stride;
    uint32_t m, src, srca;
    int32_t w;
    uint16_t t;

    __m128i alpha, tmp;
    __m128i vmask, vmask_lo, vmask_hi;
    __m128i vdst, vdst_lo, vdst_hi;
    __m128i mask_zero = __lsx_vldi(0);

    PIXMAN_IMAGE_GET_LINE(dest_image, dest_x, dest_y, uint8_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE(mask_image, mask_x, mask_y, uint8_t, mask_stride, mask_line, 1);

    src = _pixman_image_get_solid(imp, src_image, dest_image->bits.format);
    srca = src >> 24;
    alpha = __lsx_vreplgr2vr_w(src);
    alpha = __lsx_vilvl_b(mask_zero, alpha);
    alpha = __lsx_vshuf4i_w(alpha, 0x44);
    alpha = __lsx_vshuf4i_h(alpha, 0xff);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        mask = mask_line;
        mask_line += mask_stride;
        w = width;

        while (w >= 16) {
            vmask = __lsx_vld(mask, 0);
            vdst = __lsx_vld(dst, 0);
            mask += 16;
            w -= 16;

            vmask_lo = __lsx_vsllwil_hu_bu(vmask, 0);
            vmask_hi = __lsx_vexth_hu_bu(vmask);
            vdst_lo = __lsx_vsllwil_hu_bu(vdst, 0);
            vdst_hi = __lsx_vexth_hu_bu(vdst);
            vmask_lo = lsx_pix_multiply(alpha, vmask_lo);
            vmask_hi = lsx_pix_multiply(alpha, vmask_hi);
            vdst_lo = lsx_pix_multiply(vmask_lo, vdst_lo);
            vdst_hi = lsx_pix_multiply(vmask_hi, vdst_hi);
            vdst_lo = __lsx_vsat_bu(vdst_lo, 7);
            vdst_hi = __lsx_vsat_bu(vdst_hi, 7);
            tmp = __lsx_vpickev_b(vdst_hi, vdst_lo);
            __lsx_vst(tmp, dst, 0);
            dst += 16;
        }

        while (w--) {
            m = *mask++;
            m = MUL_UN8(m, srca, t);
            if (m == 0)
                *dst = 0;
            else if (m != 0xff)
                *dst = MUL_UN8(m, *dst, t);
            dst++;
        }
    }
}

static void
lsx_composite_in_8_8 (pixman_implementation_t *imp,
                      pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint8_t *dst_line, *dst;
    uint8_t *src_line, *src;
    int src_stride, dst_stride;
    int32_t w, s;
    uint16_t t;

    __m128i tmp;
    __m128i vsrc, vsrc_lo, vsrc_hi;
    __m128i vdst, vdst_lo, vdst_hi;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint8_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint8_t, src_stride, src_line, 1);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        src = src_line;
        src_line += src_stride;
        w = width;

        while (w >= 16) {
            vsrc = __lsx_vld(src, 0);
            vdst = __lsx_vld(dst, 0);
            src += 16;
            w -= 16;

            vsrc_lo = __lsx_vsllwil_hu_bu(vsrc, 0);
            vsrc_hi = __lsx_vexth_hu_bu(vsrc);
            vdst_lo = __lsx_vsllwil_hu_bu(vdst, 0);
            vdst_hi = __lsx_vexth_hu_bu(vdst);
            vdst_lo = lsx_pix_multiply(vsrc_lo, vdst_lo);
            vdst_hi = lsx_pix_multiply(vsrc_hi, vdst_hi);
            vdst_lo = __lsx_vsat_bu(vdst_lo, 7);
            vdst_hi = __lsx_vsat_bu(vdst_hi, 7);
            tmp = __lsx_vpickev_b(vdst_hi, vdst_lo);
            __lsx_vst(tmp, dst, 0);
            dst += 16;
        }

        while (w--) {
            s = *src++;
            if (s == 0)
                *dst = 0;
            else if (s != 0xff)
                *dst = MUL_UN8(s, *dst, t);
            dst++;
        }
    }
}

static void
lsx_composite_over_n_8888_8888_ca (pixman_implementation_t *imp,
                                   pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t src, srca, ns;
    uint32_t *dst_line, *dst, nd;
    uint32_t *mask_line, *mask, ma;
    int dst_stride, mask_stride;
    int32_t w;

    __m128i d, m, t;
    __m128i s, sa, d0, d1, m0, m1, t0, t1;
    __m128i zero = __lsx_vldi(0);
    __m128i bit_set = __lsx_vreplgr2vr_h(0xff);
    src = _pixman_image_get_solid (imp, src_image, dest_image->bits.format);
    srca = src >> 24;
    if (src == 0)
        return;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (mask_image, mask_x, mask_y, uint32_t, mask_stride, mask_line, 1);
    s = __lsx_vreplgr2vr_w(src);
    s = __lsx_vilvl_b(zero, s);
    sa = __lsx_vshuf4i_h(s, 0xff);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        mask = mask_line;
        mask_line += mask_stride;
        w = width;

        while (w && ((uintptr_t)dst & 15)) {
            ma = *mask++;
            if (ma == 0xffffffff) {
                if (srca == 0xff)
                    *dst = src;
                else
                    *dst = over(src, *dst);
            } else if (ma) {
                nd = *dst;
                ns = src;

                UN8x4_MUL_UN8x4(ns, ma);
                UN8x4_MUL_UN8(ma, srca);
                ma = ~ma;
                UN8x4_MUL_UN8x4_ADD_UN8x4(nd, ma, ns);

                *dst = nd;
            }
            dst++;
            w--;
        }

        while (w >= 4) {
            m = __lsx_vld(mask, 0);
            mask += 4;
            w -= 4;

            if (__lsx_bnz_v(m)) {
                d = __lsx_vld(dst, 0);
                d0 = __lsx_vsllwil_hu_bu(d, 0);
                d1 = __lsx_vexth_hu_bu(d);
                m0 = __lsx_vsllwil_hu_bu(m, 0);
                m1 = __lsx_vexth_hu_bu(m);

                t0 = lsx_pix_multiply(s, m0);
                t1 = lsx_pix_multiply(s, m1);

                m0 = lsx_pix_multiply(m0, sa);
                m1 = lsx_pix_multiply(m1, sa);
                m0 = __lsx_vxor_v(m0, bit_set);
                m1 = __lsx_vxor_v(m1, bit_set);
                d0 = lsx_pix_multiply(d0, m0);
                d1 = lsx_pix_multiply(d1, m1);

                d = __lsx_vpickev_b(d1, d0);
                t = __lsx_vpickev_b(t1, t0);
                d = __lsx_vsadd_bu(d, t);
                __lsx_vst(d, dst, 0);
            }
            dst += 4;
        }

	while (w--) {
            ma = *mask++;
            if (ma == 0xffffffff) {
                if (srca == 0xff)
                    *dst = src;
                else
                    *dst = over(src, *dst);
            } else if (ma) {
                nd = *dst;
                ns = src;

                UN8x4_MUL_UN8x4(ns, ma);
                UN8x4_MUL_UN8(ma, srca);
                ma = ~ma;
                UN8x4_MUL_UN8x4_ADD_UN8x4(nd, ma, ns);

                *dst = nd;
            }
            dst++;
        }
    }
}

static void
lsx_composite_over_reverse_n_8888 (pixman_implementation_t *imp,
                                   pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t src;
    uint32_t *dst_line, *dst;
    int dst_stride;
    int32_t w;

    __m128i d, t;
    __m128i s, d0, d1;
    __m128i zero = __lsx_vldi(0);
    __m128i bit_set = __lsx_vreplgr2vr_h(0xff);
    src = _pixman_image_get_solid (imp, src_image, dest_image->bits.format);
    if (src == 0)
        return;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);
    s = __lsx_vreplgr2vr_w(src);
    s = __lsx_vilvl_b(zero, s);

    while (height--)
    {
        dst = dst_line;
        dst_line += dst_stride;
        w = width;

        while (w && ((uintptr_t)dst & 15)) {
            d = __lsx_vldrepl_w(dst, 0);
            d0 = __lsx_vsllwil_hu_bu(d, 0);
            d0 = __lsx_vshuf4i_h(d0, 0xff);
            d0 = __lsx_vxor_v(d0, bit_set);
            d0 = lsx_pix_multiply(d0, s);
            t = __lsx_vpickev_b(zero, d0);
            d = __lsx_vsadd_bu(d, t);
            __lsx_vstelm_w(d, dst, 0, 0);
            dst += 1;
            w--;
        }

        while (w >= 4) {
            d = __lsx_vld(dst, 0);
            w -= 4;

            d0 = __lsx_vsllwil_hu_bu(d, 0);
            d1 = __lsx_vexth_hu_bu(d);
            d0 = __lsx_vshuf4i_h(d0, 0xff);
            d1 = __lsx_vshuf4i_h(d1, 0xff);
            d0 = __lsx_vxor_v(d0, bit_set);
            d1 = __lsx_vxor_v(d1, bit_set);
            d0 = lsx_pix_multiply(d0, s);
            d1 = lsx_pix_multiply(d1, s);
            t = __lsx_vpickev_b(d1, d0);
            d = __lsx_vsadd_bu(d, t);
            __lsx_vst(d, dst, 0);
            dst += 4;
        }

        while (w--) {
            d = __lsx_vldrepl_w(dst, 0);
            d0 = __lsx_vsllwil_hu_bu(d, 0);
            d0 = __lsx_vshuf4i_h(d0, 0xff);
            d0 = __lsx_vxor_v(d0, bit_set);
            d0 = lsx_pix_multiply(d0, s);
            t = __lsx_vpickev_b(zero, d0);
            d = __lsx_vsadd_bu(d, t);
            __lsx_vstelm_w(d, dst, 0, 0);
            dst += 1;
        }
    }
}

static void
lsx_composite_src_x888_8888 (pixman_implementation_t *imp,
                             pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t *dst_line, *dst;
    uint32_t *src_line, *src;
    int32_t w;
    int dst_stride, src_stride;
    __m128i mask = __lsx_vreplgr2vr_w(0xff000000);
    __m128i vsrc0, vsrc1, vsrc2, vsrc3, vsrc4, vsrc5, vsrc6, vsrc7;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint32_t, src_stride, src_line, 1);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        src = src_line;
        src_line += src_stride;
        w = width;

        while (w && ((uintptr_t)dst & 15)) {
            *dst++ = *src++ | 0xff000000;
            w--;
        }

        while (w >= 32) {
            vsrc0 = __lsx_vld(src, 0);
            vsrc1 = __lsx_vld(src, 16);
            vsrc2 = __lsx_vld(src, 32);
            vsrc3 = __lsx_vld(src, 48);
            vsrc4 = __lsx_vld(src, 64);
            vsrc5 = __lsx_vld(src, 80);
            vsrc6 = __lsx_vld(src, 96);
            vsrc7 = __lsx_vld(src, 112);
            vsrc0 = __lsx_vor_v(vsrc0, mask);
            vsrc1 = __lsx_vor_v(vsrc1, mask);
            vsrc2 = __lsx_vor_v(vsrc2, mask);
            vsrc3 = __lsx_vor_v(vsrc3, mask);
            vsrc4 = __lsx_vor_v(vsrc4, mask);
            vsrc5 = __lsx_vor_v(vsrc5, mask);
            vsrc6 = __lsx_vor_v(vsrc6, mask);
            vsrc7 = __lsx_vor_v(vsrc7, mask);
            __lsx_vst(vsrc0, dst, 0);
            __lsx_vst(vsrc1, dst, 16);
            __lsx_vst(vsrc2, dst, 32);
            __lsx_vst(vsrc3, dst, 48);
            __lsx_vst(vsrc4, dst, 64);
            __lsx_vst(vsrc5, dst, 80);
            __lsx_vst(vsrc6, dst, 96);
            __lsx_vst(vsrc7, dst, 112);

            src += 32;
            w   -= 32;
            dst += 32;
        }

        if (w >= 16) {
            vsrc0 = __lsx_vld(src, 0);
            vsrc1 = __lsx_vld(src, 16);
            vsrc2 = __lsx_vld(src, 32);
            vsrc3 = __lsx_vld(src, 48);
            vsrc0 = __lsx_vor_v(vsrc0, mask);
            vsrc1 = __lsx_vor_v(vsrc1, mask);
            vsrc2 = __lsx_vor_v(vsrc2, mask);
            vsrc3 = __lsx_vor_v(vsrc3, mask);
            __lsx_vst(vsrc0, dst, 0);
            __lsx_vst(vsrc1, dst, 16);
            __lsx_vst(vsrc2, dst, 32);
            __lsx_vst(vsrc3, dst, 48);

            src += 16;
            w   -= 16;
            dst += 16;
        }

        if (w >= 8) {
            vsrc0 = __lsx_vld(src, 0);
            vsrc1 = __lsx_vld(src, 16);
            vsrc0 = __lsx_vor_v(vsrc0, mask);
            vsrc1 = __lsx_vor_v(vsrc1, mask);
            __lsx_vst(vsrc0, dst, 0);
            __lsx_vst(vsrc1, dst, 16);

            src += 8;
            w   -= 8;
            dst += 8;
        }

        if (w >= 4) {
            vsrc0 = __lsx_vld(src, 0);
            vsrc0 = __lsx_vor_v(vsrc0, mask);
            __lsx_vst(vsrc0, dst, 0);

            src += 4;
            w   -= 4;
            dst += 4;
        }

        while (w--) {
            *dst++ = *src++ | 0xff000000;
        }
    }
}

static void
lsx_composite_add_n_8_8 (pixman_implementation_t *imp,
                         pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint8_t *dst_line, *dst;
    uint8_t *mask_line, *mask;
    int dst_stride, mask_stride;
    int32_t w;
    uint32_t src;
    uint16_t sa;

    __m128i d0;
    __m128i vsrc, t0, t1;
    __m128i a0, a0_l, a0_h;
    __m128i b0, b0_l, b0_h;
    __m128i zero = __lsx_vldi(0);
    __m128i one_half = __lsx_vreplgr2vr_h(0x80);
    __m128i g_shift  = __lsx_vreplgr2vr_h(8);

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint8_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (mask_image, mask_x, mask_y, uint8_t, mask_stride, mask_line, 1);

    src = _pixman_image_get_solid (imp, src_image, dest_image->bits.format);

    sa = (src >> 24);
    vsrc = __lsx_vreplgr2vr_h(sa);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        mask = mask_line;
        mask_line += mask_stride;
        w = width;

        while (w >= 16) {
            a0 = __lsx_vld(mask, 0);
            w -= 16;
            mask += 16;

            a0_l = __lsx_vsllwil_hu_bu(a0, 0);
            a0_h = __lsx_vexth_hu_bu(a0);

            a0_l = __lsx_vmadd_h(one_half, a0_l, vsrc);
            a0_h = __lsx_vmadd_h(one_half, a0_h, vsrc);

            a0_l = __lsx_vsadd_hu(__lsx_vsrl_h(a0_l, g_shift), a0_l);
            a0_h = __lsx_vsadd_hu(__lsx_vsrl_h(a0_h, g_shift), a0_h);

            a0_l = __lsx_vsrl_h(a0_l, g_shift);
            a0_h = __lsx_vsrl_h(a0_h, g_shift);

            b0 = __lsx_vld(dst, 0);
            b0_l = __lsx_vsllwil_hu_bu(b0, 0);
            b0_h = __lsx_vexth_hu_bu(b0);

            t0 = __lsx_vadd_h(a0_l, b0_l);
            t1 = __lsx_vadd_h(a0_h, b0_h);

            t0 = __lsx_vor_v(t0, __lsx_vsub_h(zero, __lsx_vsrl_h(t0, g_shift)));
            t1 = __lsx_vor_v(t1, __lsx_vsub_h(zero, __lsx_vsrl_h(t1, g_shift)));

            t0 = __lsx_vsat_hu(t0, 7);
            t1 = __lsx_vsat_hu(t1 ,7);

            d0 = __lsx_vpickev_b(t1, t0);
            __lsx_vst(d0, dst, 0);
            dst += 16;
        }

        while (w--) {
            uint16_t tmp;
            uint16_t a;
            uint32_t m, d;
            uint32_t r;

            a = *mask++;
            d = *dst;

            m = MUL_UN8 (sa, a, tmp);
            r = ADD_UN8 (m, d, tmp);

            *dst++ = r;
        }
    }
}

static void
lsx_composite_add_n_8 (pixman_implementation_t *imp,
                       pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint8_t *dst_line, *dst;
    int dst_stride;
    int32_t w;
    uint32_t src;

    __m128i vsrc, d0, d1;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint8_t, dst_stride, dst_line, 1);

    src = _pixman_image_get_solid (imp, src_image, dest_image->bits.format);
    src >>= 24;

    if (src == 0x00)
        return;

    if (src == 0xff) {
        pixman_fill (dest_image->bits.bits, dest_image->bits.rowstride,
                     8, dest_x, dest_y, width, height, 0xff);
        return;
    }

    vsrc = __lsx_vreplgr2vr_b(src);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        w = width;

        while (w && ((uintptr_t)dst & 15)) {
            d0 = __lsx_vldrepl_b(dst, 0);
            d0 = __lsx_vsadd_bu(vsrc, d0);
            __lsx_vstelm_b(d0, dst, 0, 0);
            w--;
            dst++;
        }

        while (w >= 32) {
            d0 = __lsx_vld(dst, 0);
            d1 = __lsx_vld(dst, 16);
            w -= 32;
            d0 = __lsx_vsadd_bu(vsrc, d0);
            d1 = __lsx_vsadd_bu(vsrc, d1);
            __lsx_vst(d0, dst, 0);
            __lsx_vst(d1, dst, 16);
            dst += 32;
        }

        if (w >= 16) {
            d0 = __lsx_vld(dst, 0);
            w -= 16;
            d0 = __lsx_vsadd_bu(vsrc, d0);
            __lsx_vst(d0, dst, 0);
            dst += 16;
        }

        if (w >= 8) {
            d0 = __lsx_vldrepl_d(dst, 0);
            w -= 8;
            d0 = __lsx_vsadd_bu(vsrc, d0);
            __lsx_vstelm_d(d0, dst, 0, 0);
            dst += 8;
        }

        if (w >= 4) {
            d0 = __lsx_vldrepl_w(dst, 0);
            w -= 4;
            d0 = __lsx_vsadd_bu(vsrc, d0);
            __lsx_vstelm_w(d0, dst, 0, 0);
            dst += 4;
        }

        while (w--) {
            d0 = __lsx_vldrepl_b(dst, 0);
            d0 = __lsx_vsadd_bu(vsrc, d0);
            __lsx_vstelm_b(d0, dst, 0, 0);
            dst++;
        }
    }
}

static void
lsx_composite_add_n_8888 (pixman_implementation_t *imp,
                          pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t *dst_line, *dst, src;
    int dst_stride, w;

    __m128i vsrc, d0, d1;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);

    src = _pixman_image_get_solid (imp, src_image, dest_image->bits.format);
    if (src == 0)
        return;

    if (src == ~0) {
        pixman_fill (dest_image->bits.bits, dest_image->bits.rowstride, 32,
                     dest_x, dest_y, width, height, ~0);
        return;
    }

    vsrc = __lsx_vreplgr2vr_w(src);

    while (height--) {
        w = width;

        dst = dst_line;
        dst_line += dst_stride;

        while (w && (uintptr_t)dst & 15) {
            d0 = __lsx_vldrepl_w(dst, 0);
            d0 = __lsx_vsadd_bu(vsrc, d0);
            __lsx_vstelm_w(d0, dst, 0, 0);
            dst++;
            w--;
        }

        while (w >= 8) {
            d0 = __lsx_vld(dst, 0);
            d1 = __lsx_vld(dst, 16);
            w -= 8;
            d0 = __lsx_vsadd_bu(vsrc, d0);
            d1 = __lsx_vsadd_bu(vsrc, d1);
            __lsx_vst(d0, dst, 0);
            __lsx_vst(d1, dst, 16);
            dst += 8;
        }

        while (w >= 4) {
            d0 = __lsx_vld(dst, 0);
            w -= 4;
            d0 = __lsx_vsadd_bu(vsrc, d0);
            __lsx_vst(d0, dst, 0);
            dst += 4;
        }

        while (w--) {
            d0 = __lsx_vldrepl_w(dst, 0);
            d0 = __lsx_vsadd_bu(vsrc, d0);
            __lsx_vstelm_w(d0, dst, 0, 0);
            dst++;
        }
    }
}

static force_inline __m128i
unpack_32_1x128 (uint32_t data)
{
    __m128i zero = __lsx_vldi(0);
    __m128i tmp = __lsx_vinsgr2vr_w(zero, data, 0);
    return __lsx_vilvl_b(zero, tmp);
}

static force_inline __m128i
unpack_32_2x128 (uint32_t data)
{
    __m128i tmp0, out0;
    __m128i zero = __lsx_vldi(0);
    tmp0 = __lsx_vinsgr2vr_w(tmp0, data, 0);
    tmp0 = __lsx_vpickev_d(tmp0, tmp0);
    out0 = __lsx_vilvl_b(zero, tmp0);

    return out0;
}

static force_inline __m128i
expand_pixel_32_1x128 (uint32_t data)
{
    return __lsx_vshuf4i_w(unpack_32_1x128(data), 0x44);
}

static force_inline __m128i
expand_pixel_32_2x128 (uint32_t data)
{
    return __lsx_vshuf4i_w(unpack_32_2x128(data), 0x44);
}

static force_inline __m128i
expand_alpha_1x128 (__m128i data)
{
    return __lsx_vshuf4i_h(data, 0xff);
}

static force_inline __m128i
expand_alphaa_2x128 (__m128i data)
{
    __m128i tmp0;
    tmp0 = __lsx_vshuf4i_h(data, 0xff);
    tmp0 = __lsx_vpickev_d(tmp0, tmp0);

    return tmp0;
}

static force_inline __m128i
unpack_565_to_8888 (__m128i lo)
{
    __m128i r, g, b, rb, t;
    __m128i mask_green_4x32 = __lsx_vreplgr2vr_w(0x0000fc00);
    __m128i mask_red_4x32   = __lsx_vreplgr2vr_w(0x00f80000);
    __m128i mask_blue_4x32  = __lsx_vreplgr2vr_w(0x000000f8);
    __m128i mask_565_fix_rb = __lsx_vreplgr2vr_w(0x00e000e0);
    __m128i mask_565_fix_g  = __lsx_vreplgr2vr_w(0x0000c000);

    r  = __lsx_vslli_w(lo, 8);
    r  = __lsx_vand_v(r, mask_red_4x32);
    g  = __lsx_vslli_w(lo, 5);
    g  = __lsx_vand_v(g, mask_green_4x32);
    b  = __lsx_vslli_w(lo, 3);
    b  = __lsx_vand_v(b, mask_blue_4x32);

    rb = __lsx_vor_v(r, b);
    t  = __lsx_vand_v(rb, mask_565_fix_rb);
    t  = __lsx_vsrli_w(t, 5);
    rb = __lsx_vor_v(rb, t);

    t  = __lsx_vand_v(g, mask_565_fix_g);
    t  = __lsx_vsrli_w(t, 6);
    g  = __lsx_vor_v(g, t);

    return (__lsx_vor_v(rb, g));
}

static force_inline void
unpack_128_2x128 (__m128i data, __m128i *data_lo, __m128i *data_hi)
{
    __m128i mask_zero = __lsx_vldi(0);
    *data_lo = __lsx_vilvl_b(mask_zero, data);
    *data_hi = __lsx_vilvh_b(mask_zero, data);
}

static force_inline void
unpack_565_128_4x128 (__m128i data, __m128i *data0,
                      __m128i *data1, __m128i *data2, __m128i *data3)
{
    __m128i lo, hi;
    __m128i mask_zero = __lsx_vldi(0);
    lo = __lsx_vilvl_h(mask_zero, data);
    hi = __lsx_vilvh_h(mask_zero, data);
    lo = unpack_565_to_8888(lo);
    hi = unpack_565_to_8888(hi);

    unpack_128_2x128((__m128i)lo, (__m128i*)data0, (__m128i*)data1);
    unpack_128_2x128((__m128i)hi, (__m128i*)data2, (__m128i*)data3);
}

static force_inline void
negate_2x128 (__m128i data_lo, __m128i data_hi, __m128i *neg_lo, __m128i *neg_hi)
{
    __m128i mask_00ff = __lsx_vreplgr2vr_h(0x00ff);
    *neg_lo = __lsx_vxor_v(data_lo, mask_00ff);
    *neg_hi = __lsx_vxor_v(data_hi, mask_00ff);
}

static force_inline void
over_2x128 (__m128i *src_lo, __m128i *src_hi, __m128i *alpha_lo,
            __m128i *alpha_hi, __m128i *dst_lo, __m128i *dst_hi)
{
    __m128i t1, t2;
    negate_2x128(*alpha_lo, *alpha_hi, &t1, &t2);
    *dst_lo = lsx_pix_multiply(*dst_lo, t1);
    *dst_hi = lsx_pix_multiply(*dst_hi, t2);
    *dst_lo = __lsx_vsadd_bu(*src_lo, *dst_lo);
    *dst_hi = __lsx_vsadd_bu(*src_hi, *dst_hi);
}

static force_inline __m128i
pack_2x128_128 (__m128i lo, __m128i hi)
{
    __m128i tmp0 = __lsx_vsat_bu(lo, 7);
    __m128i tmp1 = __lsx_vsat_bu(hi, 7);
    __m128i tmp2 = __lsx_vpickev_b(tmp1, tmp0);

    return tmp2;
}

static force_inline __m128i
pack_565_2x128_128 (__m128i lo, __m128i hi)
{
    __m128i data;
    __m128i r, g1, g2, b;
    __m128i mask_565_r  = __lsx_vreplgr2vr_w(0x00f80000);
    __m128i mask_565_g1 = __lsx_vreplgr2vr_w(0x00070000);
    __m128i mask_565_g2 = __lsx_vreplgr2vr_w(0x000000e0);
    __m128i mask_565_b  = __lsx_vreplgr2vr_w(0x0000001f);

    data = pack_2x128_128 (lo, hi);
    r    = __lsx_vand_v(data, mask_565_r);
    g1   = __lsx_vslli_w(data, 3) & mask_565_g1;
    g2   = __lsx_vsrli_w(data, 5) & mask_565_g2;
    b    = __lsx_vsrli_w(data, 3) & mask_565_b;

    return (((r|g1)|g2)|b);
}

static force_inline __m128i
expand565_16_1x128 (uint16_t pixel)
{
    __m128i m;
    __m128i zero = __lsx_vldi(0);

    m = __lsx_vinsgr2vr_w(m, pixel, 0);
    m = unpack_565_to_8888(m);
    m = __lsx_vilvl_b(zero, m);

    return m;
}

static force_inline uint32_t
pack_1x128_32 (__m128i data)
{
    __m128i tmp0, tmp1;
    __m128i zero = __lsx_vldi(0);

    tmp0 = __lsx_vsat_bu(data, 7);
    tmp1 = __lsx_vpickev_b(zero, tmp0);

    return (__lsx_vpickve2gr_wu(tmp1, 0));
}

static force_inline uint16_t
pack_565_32_16 (uint32_t pixel)
{
    return (uint16_t)(((pixel >> 8) & 0xf800) |
                      ((pixel >> 5) & 0x07e0) |
                      ((pixel >> 3) & 0x001f));
}

static force_inline __m128i
pack_565_4x128_128 (__m128i *v0, __m128i *v1, __m128i *v2, __m128i *v3)
{
    return pack_2x128_128(pack_565_2x128_128(*v0, *v1),
                          pack_565_2x128_128(*v2, *v3));
}

static force_inline void
expand_alpha_2x128 (__m128i data_lo, __m128i data_hi, __m128i *alpha_lo, __m128i *alpha_hi)
{
    *alpha_lo = __lsx_vshuf4i_h(data_lo, 0xff);
    *alpha_hi = __lsx_vshuf4i_h(data_hi, 0xff);
}

static force_inline void
expand_alpha_rev_2x128 (__m128i data_lo,  __m128i data_hi, __m128i *alpha_lo, __m128i *alpha_hi)
{
    *alpha_lo = __lsx_vshuf4i_h(data_lo, 0x00);
    *alpha_hi = __lsx_vshuf4i_h(data_hi, 0x00);
}

static force_inline uint16_t
composite_over_8888_0565pixel (uint32_t src, uint16_t dst)
{
    __m128i ms;
    ms = unpack_32_1x128(src);

    return pack_565_32_16(pack_1x128_32((__m128i)over_1x128((__m128i)ms,
                          (__m128i)expand_alpha_1x128((__m128i)ms), expand565_16_1x128(dst))));
}

static force_inline void
in_over_2x128 (__m128i *src_lo, __m128i *src_hi, __m128i *alpha_lo, __m128i *alpha_hi,
               __m128i *mask_lo, __m128i *mask_hi, __m128i *dst_lo, __m128i *dst_hi)
{
    __m128i s_lo, s_hi;
    __m128i a_lo, a_hi;
    s_lo = lsx_pix_multiply(*src_lo, *mask_lo);
    s_hi = lsx_pix_multiply(*src_hi, *mask_hi);
    a_lo = lsx_pix_multiply(*alpha_lo, *mask_lo);
    a_hi = lsx_pix_multiply(*alpha_hi, *mask_hi);
    over_2x128(&s_lo, &s_hi, &a_lo, &a_hi, dst_lo, dst_hi);
}

static force_inline __m128i
in_over_1x128 (__m128i *src, __m128i *alpha, __m128i *mask, __m128i *dst)
{
    return over_1x128(lsx_pix_multiply(*src, *mask),
                      lsx_pix_multiply(*alpha, *mask), *dst);
}

static force_inline __m128i
expand_alpha_rev_1x128 (__m128i data)
{
    __m128i v0 = {0x00000000, 0xffffffff};
    __m128i v_hi = __lsx_vand_v(data, v0);
    data = __lsx_vshuf4i_h(data, 0x00);
    v0 = __lsx_vnor_v(v0, v0);
    data = __lsx_vand_v(data, v0);
    data = __lsx_vor_v(data, v_hi);

    return data;
}

static void
lsx_composite_over_n_0565 (pixman_implementation_t *imp,
                           pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t src;
    uint16_t *dst_line, *dst, d;
    int32_t w;
    int dst_stride;
    __m128i vsrc, valpha;
    __m128i vdst, vdst0, vdst1, vdst2, vdst3;

    src = _pixman_image_get_solid (imp, src_image, dest_image->bits.format);

    if (src == 0)
        return;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint16_t, dst_stride, dst_line, 1);

    vsrc = expand_pixel_32_1x128(src);
    valpha = expand_alpha_1x128(vsrc);

    while (height--) {
        dst = dst_line;

        dst_line += dst_stride;
        w = width;

        while (w >= 8) {
            vdst = __lsx_vld(dst, 0);
            w -= 8;

            unpack_565_128_4x128(vdst, &vdst0, &vdst1, &vdst2, &vdst3);

            over_2x128(&vsrc, &vsrc, &valpha, &valpha, &vdst0, &vdst1);
            over_2x128(&vsrc, &vsrc, &valpha, &valpha, &vdst2, &vdst3);

            vdst = pack_565_4x128_128(&vdst0, &vdst1, &vdst2, &vdst3);
            __lsx_vst(vdst, dst, 0);
            dst += 8;
        }

        while (w--) {
            d = *dst;
            *dst++ = pack_565_32_16(pack_1x128_32(over_1x128(vsrc,valpha, expand565_16_1x128(d))));
        }
    }
}

static void
lsx_composite_over_8888_0565 (pixman_implementation_t *imp,
                              pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint16_t *dst_line, *dst, d;
    uint32_t *src_line, *src, s;
    int dst_stride, src_stride;
    int32_t w;

    __m128i valpha_lo, valpha_hi;
    __m128i vsrc, vsrc_lo, vsrc_hi;
    __m128i vdst, vdst0, vdst1, vdst2, vdst3;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint16_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint32_t, src_stride, src_line, 1);

    while (height--) {
        dst = dst_line;
        src = src_line;

        dst_line += dst_stride;
        src_line += src_stride;
        w = width;

        while (w >= 8) {
            vsrc = __lsx_vld(src, 0);
            vdst = __lsx_vld(dst, 0);

            unpack_128_2x128(vsrc, &vsrc_lo, &vsrc_hi);
            unpack_565_128_4x128(vdst, &vdst0, &vdst1, &vdst2, &vdst3);

            expand_alpha_2x128(vsrc_lo, vsrc_hi, &valpha_lo, &valpha_hi);

            vsrc = __lsx_vld(src, 16);
            over_2x128(&vsrc_lo, &vsrc_hi, &valpha_lo, &valpha_hi, &vdst0, &vdst1);

            unpack_128_2x128(vsrc, &vsrc_lo, &vsrc_hi);
            expand_alpha_2x128(vsrc_lo, vsrc_hi, &valpha_lo, &valpha_hi);
            over_2x128(&vsrc_lo, &vsrc_hi, &valpha_lo, &valpha_hi, &vdst2, &vdst3);

            __lsx_vst(pack_565_4x128_128(&vdst0, &vdst1, &vdst2, &vdst3), dst, 0);

            w -= 8;
            dst += 8;
            src += 8;
        }

        while (w--) {
            s = *src++;
            d = *dst;
            *dst++ = composite_over_8888_0565pixel(s, d);
        }
    }
}

static void
lsx_composite_over_n_8_0565 (pixman_implementation_t *imp,
                             pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t src;
    uint16_t *dst_line, *dst, d;
    uint8_t *mask_line, *p;
    uint32_t *mask;
    int dst_stride, mask_stride;
    int32_t w;
    uint32_t m;

    __m128i mask_zero = __lsx_vldi(0);
    __m128i lsx_src, lsx_alpha, lsx_mask, lsx_dest;
    __m128i vsrc, valpha;
    __m128i vmask, vmask_lo, vmask_hi;
    __m128i vdst, vdst0, vdst1, vdst2, vdst3;

    src = _pixman_image_get_solid (imp, src_image, dest_image->bits.format);

    if (src == 0)
        return;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint16_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (mask_image, mask_x, mask_y, uint8_t, mask_stride, mask_line, 1);

    lsx_src = expand_pixel_32_1x128(src);
    lsx_alpha = expand_alpha_1x128(lsx_src);

    vsrc = expand_pixel_32_2x128(src);
    valpha = expand_alphaa_2x128(vsrc);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        mask = (void*)mask_line;
        mask_line += mask_stride;
        w = width;

        while (w >= 8) {
            vdst = __lsx_vld(dst, 0);
            w -= 8;
            unpack_565_128_4x128(vdst, &vdst0, &vdst1, &vdst2, &vdst3);

	    m = *((uint32_t*)mask);
            mask += 1;

	    if (m)
	    {
		vmask = unpack_32_1x128(m);
		vmask = __lsx_vilvl_b(mask_zero, vmask);
		unpack_128_2x128(vmask, (__m128i*)&vmask_lo, (__m128i*)&vmask_hi);
		expand_alpha_rev_2x128(vmask_lo, vmask_hi, &vmask_lo, &vmask_hi);
		in_over_2x128(&vsrc, &vsrc, &valpha, &valpha, &vmask_lo, &vmask_hi,
                              &vdst0, &vdst1);
	    }

	    m = *((uint32_t*)mask);
            mask += 1;

	    if (m)
	    {
		vmask = unpack_32_1x128(m);
		vmask = __lsx_vilvl_b(mask_zero, vmask);
		unpack_128_2x128(vmask, (__m128i*)&vmask_lo, (__m128i*)&vmask_hi);
		expand_alpha_rev_2x128(vmask_lo, vmask_hi, &vmask_lo, &vmask_hi);
		in_over_2x128(&vsrc, &vsrc, &valpha, &valpha, &vmask_lo, &vmask_hi,
                              &vdst2, &vdst3);
	    }

            __lsx_vst(pack_565_4x128_128(&vdst0, &vdst1, &vdst2, &vdst3), dst, 0);

            dst += 8;
        }

        p = (void*)mask;
        while (w--) {
            m = *p++;

            if (m) {
                d = *dst;
                lsx_mask = expand_alpha_rev_1x128(unpack_32_1x128 (m));
                lsx_dest = expand565_16_1x128(d);

                *dst = pack_565_32_16(pack_1x128_32(in_over_1x128 (&lsx_src,
                                      &lsx_alpha, &lsx_mask, &lsx_dest)));
            }
            dst++;
        }
    }
}

static void
lsx_composite_over_x888_8_8888 (pixman_implementation_t *imp,
                                pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t *src, *src_line, s;
    uint32_t *dst, *dst_line, d;
    uint8_t  *mask_line, *p;
    uint32_t *mask;
    uint32_t m, w;
    int src_stride, mask_stride, dst_stride;

    __m128i mask_zero = __lsx_vldi(0);
    __m128i mask_00ff;
    __m128i mask_4x32;
    __m128i vsrc, vsrc_lo, vsrc_hi;
    __m128i vdst, vdst_lo, vdst_hi;
    __m128i vmask, vmask_lo, vmask_hi;

    mask_4x32 = __lsx_vreplgr2vr_w(0xff000000);
    mask_00ff = __lsx_vreplgr2vr_h(0x00ff);

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (mask_image, mask_x, mask_y, uint8_t, mask_stride, mask_line, 1);
    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint32_t, src_stride, src_line, 1);

    while (height--) {
        src = src_line;
        src_line += src_stride;
        dst = dst_line;
        dst_line += dst_stride;
        mask = (void*)mask_line;
        mask_line += mask_stride;
        w = width;

        while (w >= 4) {
            m = *mask;
            vsrc = __lsx_vld(src, 0);
            src += 4;
            w -= 4;
            vsrc |= mask_4x32;

            if (m == 0xffffffff) {
                __lsx_vst(vsrc, dst, 0);
            } else {
                vdst = __lsx_vld(dst, 0);
                vmask = __lsx_vilvl_b(mask_zero, unpack_32_1x128(m));

                unpack_128_2x128(vsrc, (__m128i*)&vsrc_lo, (__m128i*)&vsrc_hi);
                unpack_128_2x128(vmask, (__m128i*)&vmask_lo, (__m128i*)&vmask_hi);
                expand_alpha_rev_2x128(vmask_lo, vmask_hi, &vmask_lo, &vmask_hi);
                unpack_128_2x128(vdst, (__m128i*)&vdst_lo, (__m128i*)&vdst_hi);

                in_over_2x128(&vsrc_lo, &vsrc_hi, &mask_00ff, &mask_00ff,
                              &vmask_lo, &vmask_hi, &vdst_lo, &vdst_hi);

                __lsx_vst(pack_2x128_128(vdst_lo, vdst_hi), dst, 0);
            }
            dst += 4;
            mask += 1;
        }

        p = (void*)mask;
        while (w--) {
            m = *p++;

            if (m) {
                s = 0xff000000 | *src;

                if (m == 0xff) {
                    *dst = s;
                }
                else {
                    __m128i ma, md, ms;
                    d = *dst;
                    ma = expand_alpha_rev_1x128(unpack_32_1x128(m));
                    md = unpack_32_1x128(d);
                    ms = unpack_32_1x128(s);
                    *dst = pack_1x128_32(in_over_1x128(&ms, &mask_00ff, &ma, &md));
                }
            }
            src++;
            dst++;
        }
    }
}

static void
lsx_composite_over_8888_n_8888 (pixman_implementation_t *imp,
                                pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t *dst_line, *dst;
    uint32_t *src_line, *src;
    uint32_t mask, maska;
    int32_t w;
    int dst_stride, src_stride;

    __m128i vmask;
    __m128i vsrc, vsrc_lo, vsrc_hi;
    __m128i vdst, vdst_lo, vdst_hi;
    __m128i valpha_lo, valpha_hi;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint32_t, src_stride, src_line, 1);

    mask = _pixman_image_get_solid (imp, mask_image, PIXMAN_a8r8g8b8);
    maska = mask >> 24;
    vmask = __lsx_vreplgr2vr_h(maska);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        src = src_line;
        src_line += src_stride;
        w = width;

        while (w >= 4) {
            vsrc = __lsx_vld(src, 0);
            src += 4;
            w -= 4;

            if (__lsx_bnz_v(vsrc)) {
                vdst = __lsx_vld(dst, 0);
                unpack_128_2x128(vsrc, (__m128i*)&vsrc_lo, (__m128i*)&vsrc_hi);
                unpack_128_2x128(vdst, (__m128i*)&vdst_lo, (__m128i*)&vdst_hi);
                expand_alpha_2x128(vsrc_lo, vsrc_hi,  &valpha_lo, &valpha_hi);

                in_over_2x128(&vsrc_lo, &vsrc_hi, &valpha_lo, &valpha_hi,
                              &vmask, &vmask, &vdst_lo, &vdst_hi);

                __lsx_vst(pack_2x128_128(vdst_lo, vdst_hi), dst, 0);
            }
            dst += 4;
        }

        while (w--) {
            uint32_t s = *src++;

            if (s) {
                uint32_t d = *dst;
                __m128i ms = unpack_32_1x128(s);
                __m128i alpha = expand_alpha_1x128(ms);
                __m128i mask = vmask;
                __m128i dest = unpack_32_1x128(d);
                *dst = pack_1x128_32(in_over_1x128(&ms, &alpha, &mask, &dest));
            }
            dst++;
        }
    }
}

static void
lsx_composite_over_x888_n_8888 (pixman_implementation_t *imp,
                                pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t *dst_line, *dst;
    uint32_t *src_line, *src;
    uint32_t mask, maska;
    int dst_stride, src_stride;
    int32_t w;

    __m128i vmask, valpha, mask_4x32, mask_00ff;
    __m128i vsrc, vsrc_lo, vsrc_hi;
    __m128i vdst, vdst_lo, vdst_hi;
    __m128i zero = __lsx_vldi(0);

    mask_4x32 = __lsx_vreplgr2vr_w(0xff000000);
    mask_00ff = __lsx_vreplgr2vr_h(0x00ff);

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint32_t, src_stride, src_line, 1);

    mask = _pixman_image_get_solid (imp, mask_image, PIXMAN_a8r8g8b8);
    maska = mask >> 24;

    vmask = __lsx_vreplgr2vr_h(maska);
    valpha = mask_00ff;

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        src = src_line;
        src_line += src_stride;
        w = width;

        while (w >= 4) {
            vsrc = __lsx_vld(src, 0);
            src += 4;
            w -= 4;
            vsrc = __lsx_vor_v(vsrc, mask_4x32);
            vdst = __lsx_vld(dst, 0);

            unpack_128_2x128(vsrc, (__m128i*)&vsrc_lo, (__m128i*)&vsrc_hi);
            unpack_128_2x128(vdst, (__m128i*)&vdst_lo, (__m128i*)&vdst_hi);

            in_over_2x128(&vsrc_lo, &vsrc_hi, &valpha, &valpha,
                          &vmask, &vmask, &vdst_lo, &vdst_hi);

            __lsx_vst(pack_2x128_128(vdst_lo, vdst_hi), dst, 0);
            dst += 4;
        }

        while (w--) {
            uint32_t s = (*src++) | 0xff000000;
            uint32_t d = *dst;

            __m128i alpha, tmask;
            __m128i src = unpack_32_1x128 (s);
            __m128i dest  = unpack_32_1x128 (d);

            alpha = __lsx_vpickev_d(zero, valpha);
            tmask = __lsx_vpickev_d(zero, vmask);

            *dst = pack_1x128_32(in_over_1x128(&src,  &alpha, &tmask, &dest));

            dst++;
        }
    }
}

static void
lsx_composite_over_n_8888_0565_ca (pixman_implementation_t *imp,
                                   pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t src;
    uint16_t *dst_line, *dst, d;
    uint32_t *mask_line, *mask, m;
    int dst_stride, mask_stride;
    int w, flag;

    __m128i vsrc, valpha;
    __m128i lsx_src, lsx_alpha, lsx_mask, lsx_dest;
    __m128i vmask, vmask_lo, vmask_hi;
    __m128i vdst, vdst0, vdst1, vdst2, vdst3;

    src = _pixman_image_get_solid (imp, src_image, dest_image->bits.format);

    if (src == 0)
        return;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint16_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (mask_image, mask_x, mask_y, uint32_t, mask_stride, mask_line, 1);

    lsx_src = expand_pixel_32_1x128(src);
    lsx_alpha = expand_alpha_1x128(lsx_src);

    vsrc = expand_pixel_32_2x128(src);
    valpha = expand_alphaa_2x128(vsrc);

    while (height--) {
        mask = mask_line;
        dst = dst_line;
        mask_line += mask_stride;
        dst_line += dst_stride;
        w = width;

        while (w >= 8) {
            vmask = __lsx_vld(mask, 0);
            vdst = __lsx_vld(dst, 0);
            w -= 8;

            unpack_565_128_4x128(vdst, &vdst0, &vdst1, &vdst2, &vdst3);

            flag = __lsx_bnz_v(vmask);
            unpack_128_2x128(vmask, (__m128i*)&vmask_lo, (__m128i*)&vmask_hi);
            vmask = __lsx_vld(mask, 16);
            if (flag) {
                in_over_2x128(&vsrc, &vsrc, &valpha, &valpha, &vmask_lo, &vmask_hi,
                              &vdst0, &vdst1);
            }

            flag = __lsx_bnz_v(vmask);
            unpack_128_2x128(vmask, (__m128i*)&vmask_lo, (__m128i*)&vmask_hi);
            if (flag) {
                in_over_2x128(&vsrc, &vsrc, &valpha, &valpha, &vmask_lo, &vmask_hi,
                              &vdst2, &vdst3);
            }

            __lsx_vst(pack_565_4x128_128(&vdst0, &vdst1, &vdst2, &vdst3), dst, 0);
            dst += 8;
            mask += 8;
        }

        while (w--) {
            m = *(uint32_t *) mask;

            if (m) {
                d = *dst;
                lsx_mask = unpack_32_1x128(m);
                lsx_dest = expand565_16_1x128(d);
                *dst = pack_565_32_16(pack_1x128_32(in_over_1x128(&lsx_src, &lsx_alpha,
                                      &lsx_mask, &lsx_dest)));
            }
            dst++;
            mask++;
        }
    }
}

static uint32_t *
lsx_fetch_x8r8g8b8 (pixman_iter_t *iter, const uint32_t *mask)
{
    __m128i out0, out1, mask_4x32;
    int w = iter->width;
    uint32_t *dst = iter->buffer;
    uint32_t *src = (uint32_t *)iter->bits;
    iter->bits += iter->stride;
    mask_4x32 = __lsx_vreplgr2vr_w(0xff000000);

    while (w >= 8) {
        out0 = __lsx_vld(src, 0);
        out1 = __lsx_vld(src, 16);
        out0 = __lsx_vor_v(out0, mask_4x32);
        out1 = __lsx_vor_v(out1, mask_4x32);
        __lsx_vst(out0, dst, 0);
        __lsx_vst(out1, dst, 16);
        src += 8;
        dst += 8;
        w   -= 8;
    }

    if (w >= 4) {
        __lsx_vst(__lsx_vor_v(__lsx_vld(src, 0), mask_4x32), dst, 0);
        src += 4;
        dst += 4;
        w   -= 4;
    }

    while (w--) {
        *dst++ = (*src++) | 0xff000000;
    }

    return iter->buffer;
}

static uint32_t *
lsx_fetch_r5g6b5 (pixman_iter_t *iter, const uint32_t *mask)
{
    __m128i a, sa, s0, s1, s2, s3, s4;
    __m128i mask_red, mask_green, mask_blue;

    int w = iter->width;
    uint32_t *dst = iter->buffer;
    uint16_t *src = (uint16_t *)iter->bits;
    iter->bits += iter->stride;

    mask_red = __lsx_vreplgr2vr_h(248);
    mask_green = __lsx_vreplgr2vr_h(252);
    mask_blue = mask_red;
    a = __lsx_vreplgr2vr_h(255) << 8;

    while (w >= 4) {
        s0 = __lsx_vld(src, 0);
        src += 4;
        w   -= 4;
        //r
        s1 = __lsx_vsrli_h(s0, 8);
        s1 &= mask_red;
        s2 = __lsx_vsrli_h(s1, 5);
        s1 |= s2;

        //g
        s2 = __lsx_vsrli_h(s0, 3);
        s2 &= mask_green;
        s3 = __lsx_vsrli_h(s2, 6);
        s2 |= s3;

        //b
	s3 = s0 << 3;
        s3 &= mask_blue;
        s4 = __lsx_vsrli_h(s3, 5);
        s3 |= s4;

        //ar
        sa = a | s1;

        //gb
        s2 <<= 8;
        s2 |= s3;

        s1 = __lsx_vilvl_h(sa, s2);
        __lsx_vst(s1, dst, 0);
        dst += 4;
    }

    while (w--) {
        uint16_t s = *src++;
        *dst++ = convert_0565_to_8888(s);
    }

    return iter->buffer;
}

static uint32_t *
lsx_fetch_a8 (pixman_iter_t *iter, const uint32_t *mask)
{
    __m128i srcv;
    __m128i t0, t1, t2, t3;
    __m128i dst0, dst1;
    __m128i zero = __lsx_vldi(0);
    int w = iter->width;
    uint32_t *dst = iter->buffer;
    uint8_t *src = iter->bits;

    iter->bits += iter->stride;

    while (w >= 16) {
        srcv = __lsx_vld(src, 0);
        src += 16;
        w   -= 16;
        dst0 = __lsx_vilvl_b(srcv, zero);
        dst1 = __lsx_vilvh_b(srcv, zero);
        t0 = __lsx_vilvl_h(dst0, zero);
        t1 = __lsx_vilvh_h(dst0, zero);
        t2 = __lsx_vilvl_h(dst1, zero);
        t3 = __lsx_vilvh_h(dst1, zero);
        __lsx_vst(t0, dst, 0);
        __lsx_vst(t1, dst, 16);
        __lsx_vst(t2, dst, 32);
        __lsx_vst(t3, dst, 48);
        dst += 16;
    }

    while (w--) {
        *dst++ = *(src++) << 24;
    }

    return iter->buffer;
}

// fetch/store 8 bits
static void lsx_fetch_scanline_a8 (bits_image_t *image, int x, int y, int width,
                                   uint32_t *buffer, const uint32_t *mask)
{
    uint8_t *bits = (uint8_t *)(image->bits + y * image->rowstride);
    __m128i src;
    __m128i t0, t1;
    __m128i temp0, temp1, temp2, temp3;
    __m128i zero = __lsx_vldi(0);
    bits += x;

    while (width >= 16) {
        src = __lsx_vld(bits, 0);
        t0 = __lsx_vilvl_b(src, zero);
        t1 = __lsx_vilvh_b(src, zero);
        temp0 = __lsx_vilvl_h(t0, zero);
        temp1 = __lsx_vilvh_h(t0, zero);
        temp2 = __lsx_vilvl_h(t1, zero);
        temp3 = __lsx_vilvh_h(t1, zero);
        __lsx_vst(temp0, buffer, 0);
        __lsx_vst(temp1, buffer, 16);
        __lsx_vst(temp2, buffer, 32);
        __lsx_vst(temp3, buffer, 48);
        bits += 16, width -= 16, buffer += 16;
    }

    while (width--) {
        *buffer++ = ((*bits++) << 24);
    }
}

static void lsx_store_scanline_a8 (bits_image_t *image, int x, int y, int width,
                                   const uint32_t *values)
{
    uint8_t *dest = (uint8_t *)(image->bits + y * image->rowstride);
    __m128i src0, src1, src2, src3;
    dest += x;

    while (width >= 16) {
        src0 = __lsx_vld(values, 0);
        src1 = __lsx_vld(values, 16);
        src2 = __lsx_vld(values, 32);
        src3 = __lsx_vld(values, 48);
        src0 = __lsx_vsrli_w(src0, 24);
        src1 = __lsx_vsrli_w(src1, 24);
        src2 = __lsx_vsrli_w(src2, 24);
        src3 = __lsx_vsrli_w(src3, 24);
        src0 = __lsx_vpickev_h(src1, src0);
        src1 = __lsx_vpickev_h(src3, src2);
        src0 = __lsx_vpickev_b(src1, src0);
        __lsx_vst(src0, dest, 0);
        values += 16, width -= 16, dest += 16;
    }

    while (width >= 8) {
        src0 = __lsx_vld(values, 0);
        src1 = __lsx_vld(values, 16);
        src0 = __lsx_vsrli_w(src0, 24);
        src1 = __lsx_vsrli_w(src1, 24);
        src0 = __lsx_vpickev_h(src1, src0);
        src0 = __lsx_vpickev_b(src0, src0);
        __lsx_vstelm_d(src0, dest, 0, 0);
        values += 8; width -= 8; dest += 8;
    }

    while (width >= 4) {
        src0 = __lsx_vld(values, 0);
        src0 = __lsx_vsrli_w(src0, 24);
        src0 = __lsx_vpickev_h(src0, src0);
        src0 = __lsx_vpickev_b(src0, src0);
        __lsx_vstelm_w(src0, dest, 0, 0);
        values += 4; width -= 4; dest += 4;
    }

    while (width--) {
        *dest++ = ((*values++) >> 24);
    }
}

static void lsx_fetch_scanline_a2r2g2b2 (bits_image_t *image, int x, int y,
                                         int width, uint32_t *buffer,
                                         const uint32_t *mask)
{
    uint8_t *bits = (uint8_t *)(image->bits + y * image->rowstride);
    uint32_t pixel, pixel0, pixel1, pixel2, pixel3;

    __m128i src;
    __m128i t0, t1, t2, t3, t4, t5, t6, t7;
    __m128i mask0 = __lsx_vreplgr2vr_b(0xc0);
    __m128i mask1 = __lsx_vreplgr2vr_b(0x30);
    __m128i mask2 = __lsx_vreplgr2vr_b(0x0c);
    __m128i mask3 = __lsx_vreplgr2vr_b(0x03);
    bits += x;

    while (width >= 16) {
        src = __lsx_vld(bits, 0);
        t0 = (src & mask0); t1 = (src & mask1);
        t2 = (src & mask2); t3 = (src & mask3);
        t0 |= __lsx_vsrli_b(t0, 2), t0 |= __lsx_vsrli_b(t0, 4);
        t1 |= __lsx_vslli_b(t1, 2), t1 |= __lsx_vsrli_b(t1, 4);
        t2 |= __lsx_vsrli_b(t2, 2), t2 |= __lsx_vslli_b(t2, 4);
        t3 |= __lsx_vslli_b(t3, 2), t3 |= __lsx_vslli_b(t3, 4);
        t4 = __lsx_vilvl_b(t0, t1);
        t5 = __lsx_vilvh_b(t0, t1);
        t6 = __lsx_vilvl_b(t2, t3);
        t7 = __lsx_vilvh_b(t2, t3);
        t0 = __lsx_vilvl_h(t4, t6);
        t1 = __lsx_vilvh_h(t4, t6);
        t2 = __lsx_vilvl_h(t5, t7);
        t3 = __lsx_vilvh_h(t5, t7);
        __lsx_vst(t0, buffer, 0);
        __lsx_vst(t1, buffer, 16);
        __lsx_vst(t2, buffer, 32);
        __lsx_vst(t3, buffer, 48);
        bits += 16, width -= 16, buffer += 16;
    }

    while (width--) {
        pixel = *bits++;
        // a
        pixel0 = pixel & 192;
        pixel0 |= (pixel0 >> 2);
        pixel0 |= (pixel0 >> 4);
        pixel0 <<= 24;
        // r
        pixel1 = pixel & 48;
        pixel1 |= (pixel1 << 2);
        pixel1 |= (pixel1 >> 4);
        pixel1 <<= 16;
        // g
        pixel2 = pixel & 12;
        pixel2 |= (pixel2 >> 2);
        pixel2 |= (pixel2 << 4);
        pixel2 <<= 8;
        // b
        pixel3 = pixel & 3;
        pixel3 |= (pixel3 << 2);
        pixel3 |= (pixel3 << 4);
        *buffer++ = (pixel3 | pixel2 | pixel1 | pixel0);
    }
}

static void lsx_store_scanline_a2r2g2b2 (bits_image_t *image, int x, int y,
                                         int width, const uint32_t *values)
{
    uint8_t *dest = (uint8_t *)(image->bits + y * image->rowstride);
    __m128i src, d0;
    __m128i mask = __lsx_vreplgr2vr_b(0xc0);
    __m128i shuf = __lsx_vreplgr2vr_w(0x0F0B0703);

    dest += x;

    while (width >= 4) {
       src = __lsx_vld(values, 0);
       src = __lsx_vand_v(src, mask);
       src = __lsx_vor_v(src, __lsx_vslli_w(src, 6));
       src = __lsx_vor_v(src, __lsx_vslli_w(src, 12));
       d0 = __lsx_vshuf_b(src, src, shuf);
       __lsx_vstelm_w(d0, dest, 0, 0);
       dest += 4;
       values += 4;
       width -= 4;
    }

    while (width--) {
       uint32_t pixel = *values++;
       pixel &= 0xc0c0c0c0;
       pixel |= (pixel << 6);
       pixel |= (pixel << 12);
       pixel >>= 24;
       *dest++ = pixel;
    }
}

// fetch/store 16 bits
static void lsx_fetch_scanline_a1r5g5b5 (bits_image_t *image, int x, int y,
                                         int width, uint32_t *buffer,
                                         const uint32_t *mask)
{
    uint16_t *bits = (uint16_t *)(image->bits + y * image->rowstride);
    uint32_t pixel, pixel0, pixel1, pixel2;

    __m128i src;
    __m128i t, t0, t1, t2, t3;
    __m128i mask0 = __lsx_vreplgr2vr_h(0x001f);
    bits += x;

    while (width >= 4) {
        src  = __lsx_vld(bits, 0);
        t0   = (src & mask0);
        t0   = __lsx_vslli_h(t0, 3);
        t    = __lsx_vsrli_h(t0, 5);
        t0  |= t;
        t1   = __lsx_vsrli_h(src, 5);
        t1  &= mask0;
        t1   = __lsx_vslli_h(t1, 3);
        t    = __lsx_vsrli_h(t1, 5);
        t1  |= t;
        t2   = __lsx_vsrli_h(src, 10);
        t2  &= mask0;
        t2   = __lsx_vslli_h(t2, 3);
        t    = __lsx_vsrli_h(t2, 5);
        t2  |= t;
        t3   = __lsx_vsrli_h(src, 15);
        t    = __lsx_vslli_h(t3, 1);
        t3  |= t;
        t    = __lsx_vslli_h(t3, 2);
        t3  |= t;
        t    = __lsx_vslli_h(t3, 4);
        t3  |= t;
        t1 <<= 8;
        t0  |= t1;
        t3 <<= 8;
        t2  |= t3;
        t1 = __lsx_vilvl_h(t2, t0);
        __lsx_vst(t1, buffer, 0);
        bits += 4, width -= 4, buffer += 4;
    }

    while (width--) {
        pixel = *bits++;
        // a
        pixel0 = pixel >> 15;
        pixel0 <<= 7;
        pixel0 |= (pixel0 >> 1);
        pixel0 |= (pixel0 >> 2);
        pixel0 |= (pixel0 >> 4);
        pixel0 <<= 24;
        // r
        pixel1 = pixel >> 10;
        pixel1 &= 31;
        pixel1 <<= 3;
        pixel1 |= (pixel1 >> 5);
        pixel1 <<= 16;
        // g
        pixel2 = pixel >> 5;
        pixel2 &= 31;
        pixel2 <<= 3;
        pixel2 |= (pixel2 >> 5);
        pixel2 <<= 8;
        // b
        pixel &= 31;
        pixel <<= 3;
        pixel |= (pixel >> 5);
        *buffer++ = (pixel0 | pixel1 | pixel2 | pixel);
    }
}

static void lsx_store_scanline_a1r5g5b5 (bits_image_t *image, int x, int y,
                                         int width, const uint32_t *values)
{
    uint16_t *dest = (uint16_t *)(image->bits + y * image->rowstride);
    uint32_t pixel, pixel0, pixel1, pixel2, pixel3;
    __m128i in0, d0, t0, t1, t2, t3;
    __m128i mask0 = { 0x0000800000008000, 0x0000800000008000};
    __m128i mask1 = { 0x00007c0000007c00, 0x00007c0000007c00};
    __m128i mask2 = { 0x000003e0000003e0, 0x000003e0000003e0};
    __m128i mask3 = { 0x0000001f0000001f, 0x0000001f0000001f};
    __m128i mask4 = { 0x0006000400020000, 0x0006000400020000};

    dest += x;

    while (width >= 4) {
        in0 = __lsx_vld(values, 0);
        t0 = __lsx_vsrli_w(in0, 16);
        t1 = __lsx_vsrli_w(in0, 9);
        t2 = __lsx_vsrli_w(in0, 6);
        t3 = __lsx_vsrli_w(in0, 3);
        t0 = __lsx_vand_v(t0, mask0);
        t1 = __lsx_vand_v(t1, mask1);
        t2 = __lsx_vand_v(t2, mask2);
        t3 = __lsx_vand_v(t3, mask3);
        d0 = __lsx_vor_v(__lsx_vor_v(t0, t1), __lsx_vor_v(t2, t3));
        d0 = __lsx_vshuf_h(mask4, d0, d0);
        __lsx_vstelm_d(d0, dest, 0, 0);
        values += 4, width -= 4, dest += 4;
    }

    while (width--) {
        pixel = *values++;
        pixel0 = pixel >> 16;
        pixel1 = pixel >> 9;
        pixel2 = pixel >> 6;
        pixel3 = pixel >> 3;
        pixel0 &= 0x8000;
        pixel1 &= 0x7c00;
        pixel2 &= 0x03e0;
        pixel3 &= 0x001f;
        *dest++ = (pixel0 | pixel1 | pixel2 | pixel3);
    }
}

static void lsx_fetch_scanline_a4r4g4b4 (bits_image_t *image, int x, int y,
                                         int width, uint32_t *buffer,
                                         const uint32_t *mask)
{
    uint16_t *bits = (uint16_t *)(image->bits + y * image->rowstride);
    uint32_t pixel, pixel0, pixel1, pixel2;

    __m128i src;
    __m128i t, t0, t1, t2, t3;

    __m128i mask0 = __lsx_vreplgr2vr_h(0x000f);
    bits += x;

    while (width >= 4) {
        src  = __lsx_vld(bits, 0);
        t0   = __lsx_vsrli_h(src, 12);
        t    = (t0 << 4), t0 |= t;
        t1   = __lsx_vsrli_h(src, 8);
        t1  &= mask0, t = (t1 << 4), t1 |= t;
        t2   = __lsx_vsrli_h(src, 4);
        t2  &= mask0, t = (t2 << 4), t2 |= t;
        t3   = (src & mask0), t = (t3 << 4), t3 |= t;
        t0 <<= 8, t2 <<= 8, t0 |= t1, t2 |= t3;
        t1 = __lsx_vilvl_h(t0, t2);
        __lsx_vst(t1, buffer, 0);
        bits += 4, width -= 4, buffer += 4;
    }

    while (width--) {
        pixel = *bits++;
        // a
        pixel0   = pixel >> 12;
        pixel0  |= (pixel0 << 4);
        pixel0 <<= 24;
        // r
        pixel1   = pixel >> 8;
        pixel1  &= 15;
        pixel1  |= (pixel1 << 4);
        pixel1 <<= 16;
        // g
        pixel2   = pixel >> 4;
        pixel2  &= 15;
        pixel2  |= (pixel2 << 4);
        pixel2 <<= 8;
        // b
        pixel   &= 15;
        pixel   |= (pixel << 4);
        *buffer++ = (pixel0 | pixel1 | pixel2 | pixel);
    }
}

static void lsx_store_scanline_a4r4g4b4 (bits_image_t *image, int x, int y,
                                         int width, const uint32_t *values)
{
    uint16_t *dest = (uint16_t *)(image->bits + y * image->rowstride);
    uint32_t pixel, pixel0, pixel1;
    __m128i in0, d0, t0, t1;
    __m128i mask0 = __lsx_vreplgr2vr_h(0xf0f0);
    __m128i mask1 = __lsx_vreplgr2vr_h(0x00ff);
    __m128i mask2 = { 0x0006000400020000, 0x0006000400020000 };
    dest += x;

    while (width >= 4) {
        in0 = __lsx_vld(values, 0);
        in0 = __lsx_vand_v(in0, mask0);
        t0 = __lsx_vsrli_w(in0, 4);
        t1 = __lsx_vsrli_w(in0, 8);
        t0 = __lsx_vor_v(t0, t1);
        t0 = __lsx_vand_v(t0, mask1);
        t0 = __lsx_vor_v(t0, __lsx_vsrli_w(t0, 8));
        d0 = __lsx_vshuf_h(mask2, t0, t0);
        __lsx_vstelm_d(d0, dest, 0, 0);
        values += 4, width -= 4, dest += 4;
    }

    while (width--) {
        pixel   = *values++;
        pixel  &= 0xf0f0f0f0;
        pixel0  = (pixel >> 4);
        pixel1  = (pixel >> 8);
        pixel0 |= pixel1;
        pixel0 &= 0x00ff00ff;
        pixel0 |= (pixel0 >> 8);
        pixel0 &= 0xffff;
        *dest++ = pixel0;
    }
}

static const pixman_fast_path_t lsx_fast_paths[] =
{
    PIXMAN_STD_FAST_PATH (OVER, solid, a8, a8r8g8b8, lsx_composite_over_n_8_8888),
    PIXMAN_STD_FAST_PATH (OVER, solid, a8, x8r8g8b8, lsx_composite_over_n_8_8888),
    PIXMAN_STD_FAST_PATH (OVER, solid, a8, a8b8g8r8, lsx_composite_over_n_8_8888),
    PIXMAN_STD_FAST_PATH (OVER, solid, a8, x8b8g8r8, lsx_composite_over_n_8_8888),
    PIXMAN_STD_FAST_PATH_CA (OVER, solid, a8r8g8b8, r5g6b5, lsx_composite_over_n_8888_0565_ca),
    PIXMAN_STD_FAST_PATH_CA (OVER, solid, a8b8g8r8, b5g6r5, lsx_composite_over_n_8888_0565_ca),
    PIXMAN_STD_FAST_PATH (OVER, x8r8g8b8, solid, a8r8g8b8, lsx_composite_over_x888_n_8888),
    PIXMAN_STD_FAST_PATH (OVER, x8r8g8b8, solid, x8r8g8b8, lsx_composite_over_x888_n_8888),
    PIXMAN_STD_FAST_PATH (OVER, x8b8g8r8, solid, a8b8g8r8, lsx_composite_over_x888_n_8888),
    PIXMAN_STD_FAST_PATH (OVER, x8b8g8r8, solid, x8b8g8r8, lsx_composite_over_x888_n_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8r8g8b8, solid, a8r8g8b8, lsx_composite_over_8888_n_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8r8g8b8, solid, x8r8g8b8, lsx_composite_over_8888_n_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8b8g8r8, solid, a8b8g8r8, lsx_composite_over_8888_n_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8b8g8r8, solid, x8b8g8r8, lsx_composite_over_8888_n_8888),
    PIXMAN_STD_FAST_PATH (OVER, x8r8g8b8, a8, x8r8g8b8, lsx_composite_over_x888_8_8888),
    PIXMAN_STD_FAST_PATH (OVER, x8r8g8b8, a8, a8r8g8b8, lsx_composite_over_x888_8_8888),
    PIXMAN_STD_FAST_PATH (OVER, x8b8g8r8, a8, x8b8g8r8, lsx_composite_over_x888_8_8888),
    PIXMAN_STD_FAST_PATH (OVER, x8b8g8r8, a8, a8b8g8r8, lsx_composite_over_x888_8_8888),
    PIXMAN_STD_FAST_PATH (OVER, solid, a8, r5g6b5, lsx_composite_over_n_8_0565),
    PIXMAN_STD_FAST_PATH (OVER, solid, a8, b5g6r5, lsx_composite_over_n_8_0565),
    PIXMAN_STD_FAST_PATH (SRC, x8r8g8b8, null, a8r8g8b8, lsx_composite_src_x888_8888),
    PIXMAN_STD_FAST_PATH (SRC, x8b8g8r8, null, a8b8g8r8, lsx_composite_src_x888_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8r8g8b8, null, r5g6b5, lsx_composite_over_8888_0565),
    PIXMAN_STD_FAST_PATH (OVER, a8b8g8r8, null, b5g6r5, lsx_composite_over_8888_0565),
    PIXMAN_STD_FAST_PATH (OVER, solid, null, r5g6b5, lsx_composite_over_n_0565),
    PIXMAN_STD_FAST_PATH (OVER, solid, null, b5g6r5, lsx_composite_over_n_0565),
    PIXMAN_STD_FAST_PATH (OVER, a8r8g8b8, null, a8r8g8b8, lsx_composite_over_8888_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8r8g8b8, null, x8r8g8b8, lsx_composite_over_8888_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8b8g8r8, null, a8b8g8r8, lsx_composite_over_8888_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8b8g8r8, null, x8b8g8r8, lsx_composite_over_8888_8888),
    PIXMAN_STD_FAST_PATH (OVER, x8r8g8b8, null, x8r8g8b8, lsx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (OVER, x8b8g8r8, null, x8b8g8r8, lsx_composite_copy_area),
    PIXMAN_STD_FAST_PATH_CA (OVER, solid, a8r8g8b8, a8r8g8b8, lsx_composite_over_n_8888_8888_ca),
    PIXMAN_STD_FAST_PATH_CA (OVER, solid, a8r8g8b8, x8r8g8b8, lsx_composite_over_n_8888_8888_ca),
    PIXMAN_STD_FAST_PATH_CA (OVER, solid, a8b8g8r8, a8b8g8r8, lsx_composite_over_n_8888_8888_ca),
    PIXMAN_STD_FAST_PATH_CA (OVER, solid, a8b8g8r8, x8b8g8r8, lsx_composite_over_n_8888_8888_ca),
    PIXMAN_STD_FAST_PATH (OVER_REVERSE, solid, null, a8r8g8b8, lsx_composite_over_reverse_n_8888),
    PIXMAN_STD_FAST_PATH (OVER_REVERSE, solid, null, a8b8g8r8, lsx_composite_over_reverse_n_8888),
    PIXMAN_STD_FAST_PATH (ADD, a8, null, a8, lsx_composite_add_8_8),
    PIXMAN_STD_FAST_PATH (ADD, solid, a8, a8, lsx_composite_add_n_8_8),
    PIXMAN_STD_FAST_PATH (ADD, solid, null, a8, lsx_composite_add_n_8),
    PIXMAN_STD_FAST_PATH (ADD, solid, null, x8r8g8b8, lsx_composite_add_n_8888),
    PIXMAN_STD_FAST_PATH (ADD, solid, null, a8r8g8b8, lsx_composite_add_n_8888),
    PIXMAN_STD_FAST_PATH (ADD, solid, null, x8b8g8r8, lsx_composite_add_n_8888),
    PIXMAN_STD_FAST_PATH (ADD, solid, null, a8b8g8r8, lsx_composite_add_n_8888),
    PIXMAN_STD_FAST_PATH (ADD, a8r8g8b8, null, a8r8g8b8, lsx_composite_add_8888_8888),
    PIXMAN_STD_FAST_PATH (ADD, a8b8g8r8, null, a8b8g8r8, lsx_composite_add_8888_8888),
    PIXMAN_STD_FAST_PATH (SRC, a8r8g8b8, null, a8r8g8b8, lsx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, a8b8g8r8, null, a8b8g8r8, lsx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, a8r8g8b8, null, x8r8g8b8, lsx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, a8b8g8r8, null, x8b8g8r8, lsx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, x8r8g8b8, null, x8r8g8b8, lsx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, x8b8g8r8, null, x8b8g8r8, lsx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, b8g8r8a8, null, b8g8r8x8, lsx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, b8g8r8a8, null, b8g8r8a8, lsx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, b8g8r8x8, null, b8g8r8x8, lsx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, r5g6b5, null, r5g6b5, lsx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, b5g6r5, null, b5g6r5, lsx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, a8, null, a8, lsx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, a8r8g8b8, null, r5g6b5, lsx_composite_src_x888_0565),
    PIXMAN_STD_FAST_PATH (SRC, a8b8g8r8, null, b5g6r5, lsx_composite_src_x888_0565),
    PIXMAN_STD_FAST_PATH (SRC, x8r8g8b8, null, r5g6b5, lsx_composite_src_x888_0565),
    PIXMAN_STD_FAST_PATH (SRC, x8b8g8r8, null, b5g6r5, lsx_composite_src_x888_0565),
    PIXMAN_STD_FAST_PATH (IN, solid, a8, a8, lsx_composite_in_n_8_8),
    PIXMAN_STD_FAST_PATH (IN, a8, null, a8, lsx_composite_in_8_8),
    { PIXMAN_OP_NONE },
};

#define IMAGE_FLAGS                                                     \
    (FAST_PATH_STANDARD_FLAGS | FAST_PATH_ID_TRANSFORM |                \
     FAST_PATH_BITS_IMAGE | FAST_PATH_SAMPLES_COVER_CLIP_NEAREST)

static const pixman_iter_info_t lsx_iters[] =
{
    {
      PIXMAN_x8r8g8b8, IMAGE_FLAGS, ITER_NARROW,
      _pixman_iter_init_bits_stride, lsx_fetch_x8r8g8b8, NULL
    },
    {
      PIXMAN_r5g6b5, IMAGE_FLAGS, ITER_NARROW,
      _pixman_iter_init_bits_stride, lsx_fetch_r5g6b5, NULL
    },
    {
      PIXMAN_a8, IMAGE_FLAGS, ITER_NARROW,
      _pixman_iter_init_bits_stride, lsx_fetch_a8, NULL
    },
    { PIXMAN_null },
};

pixman_implementation_t *
_pixman_implementation_create_lsx (pixman_implementation_t *fallback)
{
    pixman_implementation_t *imp =
        _pixman_implementation_create (fallback, lsx_fast_paths);

    /* Set up function pointers */
    imp->combine_32[PIXMAN_OP_SRC] = lsx_combine_src_u;
    imp->combine_32[PIXMAN_OP_OVER] = lsx_combine_over_u;
    imp->combine_32[PIXMAN_OP_OVER_REVERSE] = lsx_combine_over_reverse_u;
    imp->combine_32[PIXMAN_OP_OUT] = lsx_combine_out_u;
    imp->combine_32[PIXMAN_OP_OUT_REVERSE] = lsx_combine_out_reverse_u;
    imp->combine_32[PIXMAN_OP_ADD] = lsx_combine_add_u;
    imp->combine_32[PIXMAN_OP_DISJOINT_SRC] = lsx_combine_src_u;
    imp->combine_32[PIXMAN_OP_CONJOINT_SRC] = lsx_combine_src_u;
    imp->combine_32[PIXMAN_OP_MULTIPLY] = lsx_combine_multiply_u;
    imp->combine_32_ca[PIXMAN_OP_SRC] = lsx_combine_src_ca;
    imp->combine_32_ca[PIXMAN_OP_OVER] = lsx_combine_over_ca;
    imp->combine_32_ca[PIXMAN_OP_OUT_REVERSE] = lsx_combine_out_reverse_ca;

    imp->blt = lsx_blt;
    imp->fill = lsx_fill;
    imp->iter_info = lsx_iters;

    return imp;
}

void setup_accessors_lsx (bits_image_t *image)
{
    if (image->format == PIXMAN_a8) { // 8 bits
        image->fetch_scanline_32 = lsx_fetch_scanline_a8;
        image->store_scanline_32 = lsx_store_scanline_a8;
    } else if (image->format == PIXMAN_a2r2g2b2) {
        image->fetch_scanline_32 = lsx_fetch_scanline_a2r2g2b2;
        image->store_scanline_32 = lsx_store_scanline_a2r2g2b2;
    } else if (image->format == PIXMAN_a1r5g5b5) { // 16 bits
        image->fetch_scanline_32 = lsx_fetch_scanline_a1r5g5b5;
        image->store_scanline_32 = lsx_store_scanline_a1r5g5b5;
    } else if (image->format == PIXMAN_a4r4g4b4) {
        image->fetch_scanline_32 = lsx_fetch_scanline_a4r4g4b4;
        image->store_scanline_32 = lsx_store_scanline_a4r4g4b4;
    }
}
