/*
 * Copyright © 2023 Loongson Technology Corporation Limited
 * Contributed by Shiyou Yin(yinshiyou-hf@loongson.cn)
 *                Lu Wang(wanglu@loongson.cn)
 *                Ding Song(songding@loongson.cn)
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

static __m256i mask_0080;
static __m256i mask_00ff;
static __m256i mask_0101;
static __m256i mask_ffff;
static __m256i mask_ff000000;
static __m256i mask_alpha;

static __m256i mask_565_r;
static __m256i mask_565_g1, mask_565_g2;
static __m256i mask_565_b;
static __m256i mask_red;
static __m256i mask_green;
static __m256i mask_blue;

static __m256i mask_565_fix_rb;
static __m256i mask_565_fix_g;

static __m256i mask_565_rb;
static __m256i mask_565_pack_multiplier;

static force_inline __m256i
create_mask_16_256 (uint16_t mask)
{
    return __lasx_xvrepli_h (mask);
}

static force_inline __m256i
create_mask_1x32_256 (uint32_t mask)
{
    return __lasx_xvreplgr2vr_w (mask);
}

static force_inline __m256i
create_mask_1x64_256 (int64_t mask)
{
    return __lasx_xvreplgr2vr_d (mask);
}

static force_inline uint32_t
over (uint32_t src, uint32_t dest)
{
    uint32_t a = ~src >> 24;

    UN8x4_MUL_UN8_ADD_UN8x4(dest, a, src);

    return dest;
}

static force_inline uint32_t
in (uint32_t x, uint8_t  y)
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
 * The lasx_pix_multiply(src, mask) implemented with the third way, and calculates
 * two sets of data each time.
 */

static force_inline __m256i
lasx_pix_multiply (__m256i src, __m256i mask)
{
    return __lasx_xvmuh_hu (__lasx_xvmadd_h(mask_0080, src, mask),
                            mask_0101);
}

static force_inline __m256i
lasx_over_u (__m256i src, __m256i dest)
{
    __m256i r1, r2, r3, t;
    __m256i rb_mask          = __lasx_xvreplgr2vr_w(0x00ff00ff);
    __m256i rb_one_half      = __lasx_xvreplgr2vr_w(0x00800080);
    __m256i rb_mask_plus_one = __lasx_xvreplgr2vr_w(0x10000100);
    __m256i a                = __lasx_xvsrli_w(__lasx_xvnor_v(src, src), 24);

    r1 = __lasx_xvand_v(dest, rb_mask);
    r1 = __lasx_xvmadd_w(rb_one_half, r1, a);
    t  = __lasx_xvand_v(rb_mask, __lasx_xvsrli_w(r1, 8));
    r1 = __lasx_xvadd_w(r1, t);
    r1 = __lasx_xvsrli_w(r1, 8);
    r1 = __lasx_xvand_v(r1, rb_mask);
    r2 = __lasx_xvand_v(src, rb_mask);

    r1 = __lasx_xvadd_w(r1, r2);
    t  = __lasx_xvand_v(rb_mask, __lasx_xvsrli_w(r1, 8));
    r1 = __lasx_xvor_v(r1, __lasx_xvsub_w(rb_mask_plus_one, t));
    r1 = __lasx_xvand_v(r1, rb_mask);

    r2 = __lasx_xvsrli_w(dest, 8);
    r2 = __lasx_xvand_v(r2, rb_mask);
    r2 = __lasx_xvmadd_w(rb_one_half, r2, a);
    t  = __lasx_xvand_v(rb_mask, __lasx_xvsrli_w(r2, 8));
    r2 = __lasx_xvadd_w(r2, t);
    r2 = __lasx_xvsrli_w(r2, 8);
    r2 = __lasx_xvand_v(r2, rb_mask);
    r3 = __lasx_xvand_v(rb_mask, __lasx_xvsrli_w(src, 8));

    r2 = __lasx_xvadd_w(r2, r3);
    t  = __lasx_xvand_v(rb_mask, __lasx_xvsrli_w(r2, 8));
    r2 = __lasx_xvor_v(r2, __lasx_xvsub_w(rb_mask_plus_one, t));
    r2 = __lasx_xvand_v(r2, rb_mask);

    t  = __lasx_xvor_v(r1, __lasx_xvslli_w(r2, 8));

    return t;
}

static force_inline __m256i
lasx_in_u (__m256i x, __m256i a)
{
    __m256i r1, r2, t;
    __m256i rb_mask     = __lasx_xvreplgr2vr_w(0xff00ff);
    __m256i rb_one_half = __lasx_xvreplgr2vr_w(0x800080);

    r1 = __lasx_xvand_v(x, rb_mask);
    r1 = __lasx_xvmadd_w(rb_one_half, r1, a);
    t  = __lasx_xvand_v(__lasx_xvsrli_w(r1, 8), rb_mask);
    r1 = __lasx_xvadd_w(r1, t);
    r1 = __lasx_xvsrli_w(r1, 8);
    r1 = __lasx_xvand_v(r1, rb_mask);
    r2 = __lasx_xvsrli_w(x, 8);

    r2 = __lasx_xvand_v(r2, rb_mask);
    r2 = __lasx_xvmadd_w(rb_one_half, r2, a);
    t  = __lasx_xvand_v(__lasx_xvsrli_w(r2, 8), rb_mask);
    r2 = __lasx_xvadd_w(r2, t);
    r2 = __lasx_xvsrli_w(r2, 8);
    r2 = __lasx_xvand_v(r2, rb_mask);

    t  = __lasx_xvor_v(r1, __lasx_xvslli_w(r2, 8));

    return t;
}

static void
lasx_combine_src_u (pixman_implementation_t *imp,
                    pixman_op_t              op,
                    uint32_t *               dest,
                    const uint32_t *         src,
                    const uint32_t *         mask,
                    int                      width)
{
    __m256i src0, mask0, dest0;
    __m256i zero = __lasx_xvldi(0);
    __m256i out0, out1, out2, out3, tmp0, tmp1;

    if (mask) {
        while (width >= 8) {
            src0 = __lasx_xvld(src, 0);
            mask0 = __lasx_xvld(mask, 0);
            tmp0 = __lasx_xvilvl_b(zero, src0);
            tmp1 = __lasx_xvilvh_b(zero, src0);
            out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            tmp0 = __lasx_xvilvl_b(zero, mask0);
            tmp1 = __lasx_xvilvh_b(zero, mask0);
            out1 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out3 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            out1 = __lasx_xvshuf4i_h(out1, 0xff);
            out3 = __lasx_xvshuf4i_h(out3, 0xff);
            out0 = lasx_pix_multiply(out0, out1);
            out2 = lasx_pix_multiply(out2, out3);
            dest0 = __lasx_xvpickev_b(out2, out0);
            dest0 = __lasx_xvpermi_d(dest0, 0xd8);
            __lasx_xvst(dest0, dest, 0);
            mask  += 8;
            width -= 8;
            src   += 8;
            dest  += 8;
        }

        for (int i = 0; i < width; ++i) {
            uint32_t s = combine_mask(src, mask, i);
            *dest++ = s;
        }
    } else {
        while (width >= 8) {
            src0 = __lasx_xvld(src, 0);
            __lasx_xvst(src0, dest, 0);
            width -= 8;
            src   += 8;
            dest  += 8;
        }

        if (width) {
            memcpy (dest, src, width * sizeof (uint32_t));
        }
    }
}

static void
lasx_combine_over_u_mask (uint32_t *dest,
                          const uint32_t *src,
                          const uint32_t *mask,
                          int width)
{
    __m256i bit_set = __lasx_xvreplgr2vr_h(0xff);
    __m256i src0, mask0, dest0, dest1;
    __m256i tmp0, tmp1;
    __m256i zero = __lasx_xvldi(0);
    __m256i out0, out1, out2, out3, out4, out5;

    while (width > 7) {
        src0 = __lasx_xvld(src, 0);
        dest0 = __lasx_xvld(dest, 0);
        mask0 = __lasx_xvld(mask, 0);
        tmp0 = __lasx_xvilvl_b(zero, src0);
        tmp1 = __lasx_xvilvh_b(zero, src0);
        out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_b(zero, mask0);
        tmp1 = __lasx_xvilvh_b(zero, mask0);
        out1 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        out3 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        out1 = __lasx_xvshuf4i_h(out1, 0xff);
        out3 = __lasx_xvshuf4i_h(out3, 0xff);
        out0 = lasx_pix_multiply(out0, out1);
        out2 = lasx_pix_multiply(out2, out3);
        out1 = __lasx_xvxor_v(out0, bit_set);
        out3 = __lasx_xvxor_v(out2, bit_set);
        out1 = __lasx_xvshuf4i_h(out1, 0xff);
        out3 = __lasx_xvshuf4i_h(out3, 0xff);
        tmp0 = __lasx_xvilvl_b(zero, dest0);
        tmp1 = __lasx_xvilvh_b(zero, dest0);
        out4 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        out5 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        out4 = lasx_pix_multiply(out4, out1);
        out5 = lasx_pix_multiply(out5, out3);

        dest0 = __lasx_xvpickev_b(out2, out0);
        dest0 = __lasx_xvpermi_d(dest0, 0xd8);
        dest1 = __lasx_xvpickev_b(out5, out4);
        dest1 = __lasx_xvpermi_d(dest1, 0xd8);
        dest0 = __lasx_xvsadd_bu(dest0, dest1);
        __lasx_xvst(dest0, dest, 0);
        width -= 8;
        mask  += 8;
        src   += 8;
        dest  += 8;
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

static force_inline __m256i
over_1x256 (__m256i src, __m256i alpha, __m256i dst)
{
    alpha = __lasx_xvxor_v(alpha, mask_00ff);
    alpha = lasx_pix_multiply(dst, alpha);
    return __lasx_xvsadd_bu (src, alpha);
}

static force_inline uint32_t
core_combine_over_u32 (uint32_t src, uint32_t dst)
{
    uint8_t a = src >> 24;

    if (a == 0xff) {
        return src;
    }
    else if (src) {
        __m256i zero = __lasx_xvldi(0);
        __m256i xr_src = __lasx_xvinsgr2vr_w (zero, src, 0);
        __m256i xr_dst = __lasx_xvinsgr2vr_w (zero, dst, 0);
        __m256i xr_alpha;
        __m256i tmp;

        xr_src = __lasx_xvilvl_b (zero, xr_src);
        xr_dst = __lasx_xvilvl_b (zero, xr_dst);
        xr_alpha = __lasx_xvshuf4i_h (xr_src, 0xff);

        tmp = __lasx_xvpickev_b (zero, over_1x256 (xr_src, xr_alpha, xr_dst));

        return __lasx_xvpickve2gr_wu (tmp, 0);
    }

    return dst;
}

static void
lasx_combine_over_u_no_mask (uint32_t *dst, const uint32_t *src, int width)
{
    __m256i zero = __lasx_xvldi(0);

    while (width >= 8) {
        __m256i xv_src, xv_dst;
        __m256i xv_src_ev, xv_src_od;
        __m256i alpha;
        __m256i xv_dst_ev, xv_dst_od;

        xv_src = __lasx_xvld(src, 0);
        xv_dst = __lasx_xvld(dst, 0);

        /* unpack src: 1x256 to 2x256 */
        xv_src_ev = __lasx_xvpackev_b(zero, xv_src);
        xv_src_od = __lasx_xvpackod_b(zero, xv_src);

        /* expand alpha */
        alpha = __lasx_xvshuf4i_h(xv_src_od, 0xf5);

        /* unpack dst: 1x256 to 2x256 */
        xv_dst_ev = __lasx_xvpackev_b(zero, xv_dst);
        xv_dst_od = __lasx_xvpackod_b(zero, xv_dst);

        xv_dst_ev = over_1x256(xv_src_ev, alpha, xv_dst_ev);
        xv_dst_od = over_1x256(xv_src_od, alpha, xv_dst_od);

        xv_dst = __lasx_xvpackev_b(xv_dst_od, xv_dst_ev);

        __lasx_xvst(xv_dst, dst, 0);
        width -= 8;
        src += 8;
        dst += 8;
    }

    while (width--) {
        uint32_t s = *src;
        uint32_t d = *dst;

        *dst = core_combine_over_u32 (s, d);

        ++src;
        ++dst;
    }
}

static void
lasx_combine_over_u (pixman_implementation_t *imp,
                     pixman_op_t              op,
                     uint32_t *               dest,
                     const uint32_t *         src,
                     const uint32_t *         mask,
                     int                      width)
{
    if (mask) {
        lasx_combine_over_u_mask (dest, src, mask, width);
    }
    else {
        lasx_combine_over_u_no_mask (dest, src, width);
    }
}


static void
lasx_combine_over_reverse_u (pixman_implementation_t *imp,
                             pixman_op_t              op,
                             uint32_t *               dest,
                             const uint32_t *         src,
                             const uint32_t *         mask,
                             int                      width)
{
    __m256i src0, mask0, dest0, dest1;
    __m256i zero = __lasx_xvldi(0);
    __m256i out0, out1, out2, out3, out4, out5;
    __m256i tmp0, tmp1;

    if (mask) {
        while (width > 7) {
            src0 = __lasx_xvld(src, 0);
            dest0 = __lasx_xvld(dest, 0);
            mask0 = __lasx_xvld(mask, 0);

            tmp0 = __lasx_xvilvl_b(zero, src0);
            tmp1 = __lasx_xvilvh_b(zero, src0);
            out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            tmp0 = __lasx_xvilvl_b(zero, mask0);
            tmp1 = __lasx_xvilvh_b(zero, mask0);
            out1 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out3 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            out1 = __lasx_xvshuf4i_h(out1, 0xff);
            out3 = __lasx_xvshuf4i_h(out3, 0xff);
            out0 = lasx_pix_multiply(out0, out1);
            out2 = lasx_pix_multiply(out2, out3);

            dest1 = __lasx_xvxori_b(dest0, 0xff);
            tmp0 = __lasx_xvilvl_b(zero, dest0);
            tmp1 = __lasx_xvilvh_b(zero, dest0);
            out1 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out3 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            tmp0 = __lasx_xvilvl_b(zero, dest1);
            tmp1 = __lasx_xvilvh_b(zero, dest1);
            out4 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out5 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            out4 = __lasx_xvshuf4i_h(out4, 0xff);
            out5 = __lasx_xvshuf4i_h(out5, 0xff);
            out0 = lasx_pix_multiply(out0, out4);
            out2 = lasx_pix_multiply(out2, out5);
            dest0 = __lasx_xvpickev_b(out2, out0);
            dest0 = __lasx_xvpermi_d(dest0, 0xd8);
            dest1 = __lasx_xvpickev_b(out3, out1);
            dest1 = __lasx_xvpermi_d(dest1, 0xd8);
            dest0 = __lasx_xvsadd_bu(dest0, dest1);
            __lasx_xvst(dest0, dest, 0);
            mask  += 8;
            width -= 8;
            src   += 8;
            dest  += 8;
        }
    } else {
        while (width > 7) {
            src0 = __lasx_xvld(src, 0);
            dest0 = __lasx_xvld(dest, 0);
            dest1 = __lasx_xvxori_b(dest0, 0xff);
            tmp0 = __lasx_xvilvl_b(zero, src0);
            tmp1 = __lasx_xvilvh_b(zero, src0);
            out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            tmp0 = __lasx_xvilvl_b(zero, dest0);
            tmp1 = __lasx_xvilvh_b(zero, dest0);
            out1 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out3 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            tmp0 = __lasx_xvilvl_b(zero, dest1);
            tmp1 = __lasx_xvilvh_b(zero, dest1);
            out4 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out5 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            out4 = __lasx_xvshuf4i_h(out4, 0xff);
            out5 = __lasx_xvshuf4i_h(out5, 0xff);
            out0 = lasx_pix_multiply(out0, out4);
            out2 = lasx_pix_multiply(out2, out5);
            dest0 = __lasx_xvpickev_b(out2, out0);
            dest0 = __lasx_xvpermi_d(dest0, 0xd8);
            dest1 = __lasx_xvpickev_b(out3, out1);
            dest1 = __lasx_xvpermi_d(dest1, 0xd8);
            dest0 = __lasx_xvsadd_bu(dest0, dest1);
            __lasx_xvst(dest0, dest, 0);
            width -= 8;
            src   += 8;
            dest  += 8;
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
lasx_combine_out_u (pixman_implementation_t *imp,
                    pixman_op_t              op,
                    uint32_t *               dest,
                    const uint32_t *         src,
                    const uint32_t *         mask,
                    int                      width)
{
    __m256i src0, mask0, dest0;
    __m256i zero = __lasx_xvldi(0);
    __m256i out0, out1, out2, out3;
    __m256i tmp0, tmp1;

    if (mask) {
        while (width > 7) {
            src0 = __lasx_xvld(src, 0);
            dest0 = __lasx_xvld(dest, 0);
            mask0 = __lasx_xvld(mask, 0);

            tmp0 = __lasx_xvilvl_b(zero, src0);
            tmp1 = __lasx_xvilvh_b(zero, src0);
            out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            tmp0 = __lasx_xvilvl_b(zero, mask0);
            tmp1 = __lasx_xvilvh_b(zero, mask0);
            out1 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out3 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            out1 = __lasx_xvshuf4i_h(out1, 0xff);
            out3 = __lasx_xvshuf4i_h(out3, 0xff);
            out0 = lasx_pix_multiply(out0, out1);
            out2 = lasx_pix_multiply(out2, out3);

            dest0 = __lasx_xvxori_b(dest0, 0xff);
            tmp0 = __lasx_xvilvl_b(zero, dest0);
            tmp1 = __lasx_xvilvh_b(zero, dest0);
            out1 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out3 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            out1 = __lasx_xvshuf4i_h(out1, 0xff);
            out3 = __lasx_xvshuf4i_h(out3, 0xff);
            out0 = lasx_pix_multiply(out0, out1);
            out2 = lasx_pix_multiply(out2, out3);
            dest0 = __lasx_xvpickev_b(out2, out0);
            dest0 = __lasx_xvpermi_d(dest0, 0xd8);
            __lasx_xvst(dest0, dest, 0);
            mask  += 8;
            width -= 8;
            src   += 8;
            dest  += 8;
        }
    } else {
        while (width > 7) {
            src0 = __lasx_xvld(src, 0);
            dest0 = __lasx_xvld(dest, 0);
            tmp0 = __lasx_xvilvl_b(zero, src0);
            tmp1 = __lasx_xvilvh_b(zero, src0);
            out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            dest0 = __lasx_xvxori_b(dest0, 0xff);
            tmp0 = __lasx_xvilvl_b(zero, dest0);
            tmp1 = __lasx_xvilvh_b(zero, dest0);
            out1 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out3 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            out1 = __lasx_xvshuf4i_h(out1, 0xff);
            out3 = __lasx_xvshuf4i_h(out3, 0xff);
            out0 = lasx_pix_multiply(out0, out1);
            out2 = lasx_pix_multiply(out2, out3);
            dest0 = __lasx_xvpickev_b(out2, out0);
            dest0 = __lasx_xvpermi_d(dest0, 0xd8);
            __lasx_xvst(dest0, dest, 0);
            width -= 8;
            src   += 8;
            dest  += 8;
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
lasx_combine_out_reverse_u (pixman_implementation_t *imp,
                            pixman_op_t              op,
                            uint32_t *               dest,
                            const uint32_t *         src,
                            const uint32_t *         mask,
                            int                      width)
{
    __m256i bit_set = __lasx_xvreplgr2vr_h(0xff);
    __m256i src0, mask0, dest0;
    __m256i zero = __lasx_xvldi(0);
    __m256i out0, out1, out2, out3;
    __m256i tmp0, tmp1;

    if (mask) {
        while (width > 7) {
            src0 = __lasx_xvld(src, 0);
            dest0 = __lasx_xvld(dest, 0);
            mask0 = __lasx_xvld(mask, 0);

            tmp0 = __lasx_xvilvl_b(zero, src0);
            tmp1 = __lasx_xvilvh_b(zero, src0);
            out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            tmp0 = __lasx_xvilvl_b(zero, mask0);
            tmp1 = __lasx_xvilvh_b(zero, mask0);
            out1 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out3 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            out1 = __lasx_xvshuf4i_h(out1, 0xff);
            out3 = __lasx_xvshuf4i_h(out3, 0xff);
            out0 = lasx_pix_multiply(out0, out1);
            out2 = lasx_pix_multiply(out2, out3);
            out1 = __lasx_xvxor_v(out0, bit_set);
            out3 = __lasx_xvxor_v(out2, bit_set);
            out1 = __lasx_xvshuf4i_h(out1, 0xff);
            out3 = __lasx_xvshuf4i_h(out3, 0xff);
            tmp0 = __lasx_xvilvl_b(zero, dest0);
            tmp1 = __lasx_xvilvh_b(zero, dest0);
            out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            out0 = lasx_pix_multiply(out0, out1);
            out2 = lasx_pix_multiply(out2, out3);
            dest0 = __lasx_xvpickev_b(out2, out0);
            dest0 = __lasx_xvpermi_d(dest0, 0xd8);
            __lasx_xvst(dest0, dest, 0);
            mask  += 8;
            width -= 8;
            src   += 8;
            dest  += 8;
        }
    } else {
        while (width > 7) {
            src0 = __lasx_xvld(src, 0);
            dest0 = __lasx_xvld(dest, 0);
            tmp0 = __lasx_xvilvl_b(zero, src0);
            tmp1 = __lasx_xvilvh_b(zero, src0);
            out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            out1 = __lasx_xvxor_v(out0, bit_set);
            out3 = __lasx_xvxor_v(out2, bit_set);
            out1 = __lasx_xvshuf4i_h(out1, 0xff);
            out3 = __lasx_xvshuf4i_h(out3, 0xff);
            tmp0 = __lasx_xvilvl_b(zero, dest0);
            tmp1 = __lasx_xvilvh_b(zero, dest0);
            out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            out0 = lasx_pix_multiply(out0, out1);
            out2 = lasx_pix_multiply(out2, out3);
            dest0 = __lasx_xvpickev_b(out2, out0);
            dest0 = __lasx_xvpermi_d(dest0, 0xd8);
            __lasx_xvst(dest0, dest, 0);
            width -= 8;
            src   += 8;
            dest  += 8;
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
lasx_combine_add_u (pixman_implementation_t *imp,
                    pixman_op_t              op,
                    uint32_t *               dest,
                    const uint32_t *         src,
                    const uint32_t *         mask,
                    int                      width)
{
    __m256i src0, mask0, dest0, dest1;
    __m256i zero = __lasx_xvldi(0);
    __m256i out0, out1, out2, out3;
    __m256i tmp0, tmp1;

    if (mask) {
        while (width > 7) {
            src0 = __lasx_xvld(src, 0);
            dest0 = __lasx_xvld(dest, 0);
            mask0 = __lasx_xvld(mask, 0);

            tmp0 = __lasx_xvilvl_b(zero, src0);
            tmp1 = __lasx_xvilvh_b(zero, src0);
            out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            tmp0 = __lasx_xvilvl_b(zero, mask0);
            tmp1 = __lasx_xvilvh_b(zero, mask0);
            out1 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out3 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            out1 = __lasx_xvshuf4i_h(out1, 0xff);
            out3 = __lasx_xvshuf4i_h(out3, 0xff);
            out0 = lasx_pix_multiply(out0, out1);
            out2 = lasx_pix_multiply(out2, out3);

            dest1 = __lasx_xvpickev_b(out2, out0);
            dest1 = __lasx_xvpermi_d(dest1, 0xd8);
            dest0 = __lasx_xvsadd_bu(dest0, dest1);
            __lasx_xvst(dest0, dest, 0);
            mask  += 8;
            width -= 8;
            src   += 8;
            dest  += 8;
        }
    } else {
        while (width > 7) {
            src0 = __lasx_xvld(src, 0);
            dest0 = __lasx_xvld(dest, 0);
            tmp0 = __lasx_xvilvl_b(zero, src0);
            tmp1 = __lasx_xvilvh_b(zero, src0);
            out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            dest1 = __lasx_xvpickev_b(out2, out0);
            dest1 = __lasx_xvpermi_d(dest1, 0xd8);
            dest0 = __lasx_xvsadd_bu(dest0, dest1);
            __lasx_xvst(dest0, dest, 0);
            width -= 8;
            src   += 8;
            dest  += 8;
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
lasx_combine_multiply_u (pixman_implementation_t *imp,
                         pixman_op_t              op,
                         uint32_t *               dest,
                         const uint32_t *         src,
                         const uint32_t *         mask,
                         int                      width)
{
    __m256i bit_set = __lasx_xvreplgr2vr_h(0xff);
    __m256i src0, mask0, dest0, dest1;
    __m256i zero = __lasx_xvldi(0);
    __m256i out0, out1, out2, out3, out4, out5, out6, out7;
    __m256i tmp0, tmp1;

    if (mask) {
        while (width > 7) {
            src0 = __lasx_xvld(src, 0);
            dest0 = __lasx_xvld(dest, 0);
            mask0 = __lasx_xvld(mask, 0);

            tmp0 = __lasx_xvilvl_b(zero, src0);
            tmp1 = __lasx_xvilvh_b(zero, src0);
            out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            tmp0 = __lasx_xvilvl_b(zero, mask0);
            tmp1 = __lasx_xvilvh_b(zero, mask0);
            out1 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out3 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            out1 = __lasx_xvshuf4i_h(out1, 0xff);
            out3 = __lasx_xvshuf4i_h(out3, 0xff);
            out0 = lasx_pix_multiply(out0, out1);
            out2 = lasx_pix_multiply(out2, out3);

            out1 = __lasx_xvxor_v(out0, bit_set);
            out3 = __lasx_xvxor_v(out2, bit_set);
            out1 = __lasx_xvshuf4i_h(out1, 0xff);
            out3 = __lasx_xvshuf4i_h(out3, 0xff);
            dest1 = __lasx_xvxori_b(dest0, 0xff);
            dest1 = __lasx_xvshuf4i_b(dest1, 0xff);
            tmp0 = __lasx_xvilvl_b(zero, dest0);
            tmp1 = __lasx_xvilvh_b(zero, dest0);
            out4 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out5 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            tmp0 = __lasx_xvilvl_b(zero, dest1);
            tmp1 = __lasx_xvilvh_b(zero, dest1);
            out6 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out7 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            out6 = lasx_pix_multiply(out0, out6);
            out7 = lasx_pix_multiply(out2, out7);
            out1 = lasx_pix_multiply(out4, out1);
            out3 = lasx_pix_multiply(out5, out3);
            dest0 = __lasx_xvpickev_b(out7, out6);
            dest0 = __lasx_xvpermi_d(dest0, 0xd8);
            dest1 = __lasx_xvpickev_b(out3, out1);
            dest1 = __lasx_xvpermi_d(dest1, 0xd8);
            dest0 = __lasx_xvsadd_bu(dest0, dest1);

            out4 = lasx_pix_multiply(out4, out0);
            out5 = lasx_pix_multiply(out5, out2);
            dest1 = __lasx_xvpickev_b(out5, out4);
            dest1 = __lasx_xvpermi_d(dest1, 0xd8);
            dest0 = __lasx_xvsadd_bu(dest0, dest1);
            __lasx_xvst(dest0, dest, 0);
            mask  += 8;
            width -= 8;
            src   += 8;
            dest  += 8;
        }
    } else {
        while (width > 7) {
            src0 = __lasx_xvld(src, 0);
            dest0 = __lasx_xvld(dest, 0);
            tmp0 = __lasx_xvilvl_b(zero, src0);
            tmp1 = __lasx_xvilvh_b(zero, src0);
            out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            out1 = __lasx_xvxor_v(out0, bit_set);
            out3 = __lasx_xvxor_v(out2, bit_set);
            out1 = __lasx_xvshuf4i_h(out1, 0xff);
            out3 = __lasx_xvshuf4i_h(out3, 0xff);
            dest1 = __lasx_xvxori_b(dest0, 0xff);
            dest1 = __lasx_xvshuf4i_b(dest1, 0xff);
            tmp0 = __lasx_xvilvl_b(zero, dest0);
            tmp1 = __lasx_xvilvh_b(zero, dest0);
            out4 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out5 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            tmp0 = __lasx_xvilvl_b(zero, dest1);
            tmp1 = __lasx_xvilvh_b(zero, dest1);
            out6 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
            out7 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
            out6 = lasx_pix_multiply(out0, out6);
            out7 = lasx_pix_multiply(out2, out7);
            out1 = lasx_pix_multiply(out4, out1);
            out3 = lasx_pix_multiply(out5, out3);
            dest0 = __lasx_xvpickev_b(out7, out6);
            dest0 = __lasx_xvpermi_d(dest0, 0xd8);
            dest1 = __lasx_xvpickev_b(out3, out1);
            dest1 = __lasx_xvpermi_d(dest1, 0xd8);
            dest0 = __lasx_xvsadd_bu(dest0, dest1);

            out4 = lasx_pix_multiply(out4, out0);
            out5 = lasx_pix_multiply(out5, out2);
            dest1 = __lasx_xvpickev_b(out5, out4);
            dest1 = __lasx_xvpermi_d(dest1, 0xd8);
            dest0 = __lasx_xvsadd_bu(dest0, dest1);
            __lasx_xvst(dest0, dest, 0);
            width -= 8;
            src   += 8;
            dest  += 8;
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
lasx_combine_src_ca (pixman_implementation_t *imp,
                     pixman_op_t              op,
                     uint32_t *               dest,
                     const uint32_t *         src,
                     const uint32_t *         mask,
                     int                      width)
{
    __m256i src0, mask0, dest0;
    __m256i zero = __lasx_xvldi(0);
    __m256i out0, out1, out2, out3;
    __m256i tmp0, tmp1;

    while (width > 7) {
        src0 = __lasx_xvld(src, 0);
        mask0 = __lasx_xvld(mask, 0);
        tmp0 = __lasx_xvilvl_b(zero, src0);
        tmp1 = __lasx_xvilvh_b(zero, src0);
        out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_b(zero, mask0);
        tmp1 = __lasx_xvilvh_b(zero, mask0);
        out1 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        out3 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        out0 = lasx_pix_multiply(out0, out1);
        out2 = lasx_pix_multiply(out2, out3);
        dest0 = __lasx_xvpickev_b(out2, out0);
        dest0 = __lasx_xvpermi_d(dest0, 0xd8);
        __lasx_xvst(dest0, dest, 0);
        mask  += 8;
        width -= 8;
        src   += 8;
        dest  += 8;
    }

    for (int i = 0; i < width; ++i) {
        uint32_t s = *(src + i);
        uint32_t m = *(mask + i);
        combine_mask_value_ca(&s, &m);
        *(dest + i) = s;
    }
}

static void
lasx_combine_over_ca (pixman_implementation_t  *imp,
                      pixman_op_t               op,
                      uint32_t *                dest,
                      const uint32_t *          src,
                      const uint32_t *          mask,
                      int                       width)
{
    __m256i bit_set = __lasx_xvreplgr2vr_h(0xff);
    __m256i src0, mask0, dest0, dest1;
    __m256i zero = __lasx_xvldi(0);
    __m256i out0, out1, out2, out3, out4, out5;
    __m256i tmp0, tmp1;

    while (width > 7) {
        src0 = __lasx_xvld(src, 0);
        dest0 = __lasx_xvld(dest, 0);
        mask0 = __lasx_xvld(mask, 0);

        tmp0 = __lasx_xvilvl_b(zero, src0);
        tmp1 = __lasx_xvilvh_b(zero, src0);
        out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_b(zero, mask0);
        tmp1 = __lasx_xvilvh_b(zero, mask0);
        out1 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        out3 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        out4 = lasx_pix_multiply(out0, out1);
        out5 = lasx_pix_multiply(out2, out3);
        out0 = __lasx_xvshuf4i_h(out0, 0xff);
        out2 = __lasx_xvshuf4i_h(out2, 0xff);
        out1 = lasx_pix_multiply(out1, out0);
        out3 = lasx_pix_multiply(out3, out2);

        out1 = __lasx_xvxor_v(out1, bit_set);
        out3 = __lasx_xvxor_v(out3, bit_set);
        tmp0 = __lasx_xvilvl_b(zero, dest0);
        tmp1 = __lasx_xvilvh_b(zero, dest0);
        out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        out1 = lasx_pix_multiply(out1, out0);
        out3 = lasx_pix_multiply(out3, out2);

        dest0 = __lasx_xvpickev_b(out5, out4);
        dest0 = __lasx_xvpermi_d(dest0, 0xd8);
        dest1 = __lasx_xvpickev_b(out3, out1);
        dest1 = __lasx_xvpermi_d(dest1, 0xd8);
        dest0 = __lasx_xvsadd_bu(dest0, dest1);
        __lasx_xvst(dest0, dest, 0);
        mask  += 8;
        width -= 8;
        src   += 8;
        dest  += 8;
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
lasx_combine_out_reverse_ca (pixman_implementation_t *imp,
                             pixman_op_t              op,
                             uint32_t *               dest,
                             const uint32_t *         src,
                             const uint32_t *         mask,
                             int                      width)
{
    __m256i bit_set = __lasx_xvreplgr2vr_h(0xff);
    __m256i src0, mask0, dest0;
    __m256i zero = __lasx_xvldi(0);
    __m256i out0, out1, out2, out3;
    __m256i tmp0, tmp1;

    while (width > 7) {
        src0 = __lasx_xvld(src, 0);
        dest0 = __lasx_xvld(dest, 0);
        mask0 = __lasx_xvld(mask, 0);

        tmp0 = __lasx_xvilvl_b(zero, src0);
        tmp1 = __lasx_xvilvh_b(zero, src0);
        out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_b(zero, mask0);
        tmp1 = __lasx_xvilvh_b(zero, mask0);
        out1 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        out3 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        out0 = __lasx_xvshuf4i_h(out0, 0xff);
        out2 = __lasx_xvshuf4i_h(out2, 0xff);
        out1 = lasx_pix_multiply(out1, out0);
        out3 = lasx_pix_multiply(out3, out2);

        out1 = __lasx_xvxor_v(out1, bit_set);
        out3 = __lasx_xvxor_v(out3, bit_set);
        tmp0 = __lasx_xvilvl_b(zero, dest0);
        tmp1 = __lasx_xvilvh_b(zero, dest0);
        out0 = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        out2 = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        out1 = lasx_pix_multiply(out1, out0);
        out3 = lasx_pix_multiply(out3, out2);
        dest0 = __lasx_xvpickev_b(out3, out1);
        dest0 = __lasx_xvpermi_d(dest0, 0xd8);
        __lasx_xvst(dest0, dest, 0);
        mask  += 8;
        width -= 8;
        src   += 8;
        dest  += 8;
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
lasx_blt_one_line_u8 (uint8_t *pDst, uint8_t *pSrc, int w)
{
    while (((uintptr_t)pDst & 31) && w) {
        *pDst = *pSrc;
        pSrc += 1;
        pDst += 1;
        w -= 1;
    }

    while (w >= 64) {
        __m256i src0, src1;
        src0 = __lasx_xvld(pSrc, 0);
        src1 = __lasx_xvld(pSrc, 32);
        __lasx_xvst(src0, pDst, 0);
        __lasx_xvst(src1, pDst, 32);

        w -= 64;
        pSrc += 64;
        pDst += 64;
    }

    if (w >= 32) {
        __lasx_xvst(__lasx_xvld(pSrc, 0), pDst, 0);

        w -= 32;
        pSrc += 32;
        pDst += 32;
    }

    while (w >= 8) {
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
lasx_blt_one_line_u16 (uint16_t *pDst, uint16_t *pSrc, int w)
{
    /* align the dst to 32 byte */
    while (((uintptr_t)pDst & 31) && w) {
        *pDst++ = *pSrc++;
        --w;
    }

    while (w >= 64) {
        /* copy 128 bytes */
        __m256i src0, src1, src2, src3;

        src0 = __lasx_xvld(pSrc, 0);
        src1 = __lasx_xvld(pSrc, 32);
        src2 = __lasx_xvld(pSrc, 64);
        src3 = __lasx_xvld(pSrc, 96);

        __lasx_xvst(src0, pDst, 0);
        __lasx_xvst(src1, pDst, 32);
        __lasx_xvst(src2, pDst, 64);
        __lasx_xvst(src3, pDst, 96);

        w -= 64;
        pSrc += 64;
        pDst += 64;
    }

    if (w >= 32) {
        /* copy 64 bytes */
        __m256i src0, src1;

        src0 = __lasx_xvld(pSrc, 0);
        src1 = __lasx_xvld(pSrc, 32);

        __lasx_xvst(src0, pDst, 0);
        __lasx_xvst(src1, pDst, 32);

        w -= 32;
        pSrc += 32;
        pDst += 32;
    }

    if (w >= 16) {
        /* copy 32 bytes */
        __lasx_xvst(__lasx_xvld(pSrc, 0), pDst, 0);

        w -= 16;
        pSrc += 16;
        pDst += 16;
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
lasx_blt_one_line_u32 (uint32_t *pDst, uint32_t *pSrc, int w)
{
    /* align the dst to 32 byte */
    while (((uintptr_t)pDst & 31) && w) {
        *pDst++ = *pSrc++;
        --w;
    }

    while (w >= 64) {
        __m256i src0, src1, src2, src3;
        __m256i src4, src5, src6, src7;

        src0 = __lasx_xvld(pSrc, 0);
        src1 = __lasx_xvld(pSrc, 32);
        src2 = __lasx_xvld(pSrc, 64);
        src3 = __lasx_xvld(pSrc, 96);
        src4 = __lasx_xvld(pSrc, 128);
        src5 = __lasx_xvld(pSrc, 160);
        src6 = __lasx_xvld(pSrc, 192);
        src7 = __lasx_xvld(pSrc, 224);

        __lasx_xvst(src0, pDst, 0);
        __lasx_xvst(src1, pDst, 32);
        __lasx_xvst(src2, pDst, 64);
        __lasx_xvst(src3, pDst, 96);
        __lasx_xvst(src4, pDst, 128);
        __lasx_xvst(src5, pDst, 160);
        __lasx_xvst(src6, pDst, 192);
        __lasx_xvst(src7, pDst, 224);

        w -= 64;
        pSrc += 64;
        pDst += 64;
    }

    if (w >= 32) {
        /* copy 32 bytes once a time */
        __m256i src0, src1, src2, src3;

        src0 = __lasx_xvld(pSrc, 0);
        src1 = __lasx_xvld(pSrc, 32);
        src2 = __lasx_xvld(pSrc, 64);
        src3 = __lasx_xvld(pSrc, 96);

        __lasx_xvst(src0, pDst, 0);
        __lasx_xvst(src1, pDst, 32);
        __lasx_xvst(src2, pDst, 64);
        __lasx_xvst(src3, pDst, 96);

        w -= 32;
        pSrc += 32;
        pDst += 32;
    }

    if (w >= 16) {
        /* copy 64 bytes once a time */
        __m256i src0, src1;

        src0 = __lasx_xvld(pSrc, 0);
        src1 = __lasx_xvld(pSrc, 32);

        __lasx_xvst(src0, pDst, 0);
        __lasx_xvst(src1, pDst, 32);

        w -= 16;
        pSrc += 16;
        pDst += 16;
    }

    if (w >= 8) {
        __m256i src;
        /* copy 32 bytes once a time */
        src = __lasx_xvld(pSrc, 0);
        __lasx_xvst(src, pDst, 0);

        w -= 8;
        pSrc += 8;
        pDst += 8;
    }

    while (w--) {
        /* copy 4 bytes once a time */
        *pDst++ = *pSrc++;
    }
}

static pixman_bool_t
lasx_blt (pixman_implementation_t *imp,
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
            lasx_blt_one_line_u8 (dst_b, src_b, width);
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
            lasx_blt_one_line_u16 (dst_h, src_h, width);
            dst_h += dst_stride;
            src_h += src_stride;
        }

        return TRUE;
    }

    if (src_bpp == 32) {
        src_bits += src_stride * src_y + src_x;
        dst_bits += dst_stride * dest_y + dest_x;

        while (height--) {
            lasx_blt_one_line_u32 (dst_bits, src_bits, width);
            dst_bits += dst_stride;
            src_bits += src_stride;
        }

        return TRUE;
    }

    return FALSE;
}

static void
lasx_fill_u8 (uint8_t  *dst,
              int       stride,
              int       x,
              int       y,
              int       width,
              int       height,
              uint8_t   filler)
{
    __m256i xvfill = __lasx_xvreplgr2vr_b(filler);
    int byte_stride = stride * 4;
    dst += y * byte_stride + x;

    while (height--) {
        int w = width;
        uint8_t *d = dst;

        while (w && ((uintptr_t)d & 31)) {
            *d = filler;
            w--;
            d++;
        }

        while (w >= 64) {
            __lasx_xvst(xvfill, d, 0);
            __lasx_xvst(xvfill, d, 32);
            w -= 64;
            d += 64;
        }

        if (w >= 32) {
            __lasx_xvst(xvfill, d, 0);
            w -= 32;
            d += 32;
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
lasx_fill_u16 (uint16_t *dst,
               int       stride,
               int       x,
               int       y,
               int       width,
               int       height,
               uint16_t  filler)
{
    __m256i xvfill = __lasx_xvreplgr2vr_h(filler);
    int short_stride = stride * 2;
    dst += y * short_stride + x;

    while (height--) {
        int w = width;
        uint16_t *d = dst;

        while (w && ((uintptr_t)d & 31)) {
            *d = filler;
            w--;
            d++;
        }

        while (w >= 32) {
            __lasx_xvst(xvfill, d, 0);
            __lasx_xvst(xvfill, d, 32);
            w -= 32;
            d += 32;
        }

        if (w >= 16) {
            __lasx_xvst(xvfill, d, 0);
            w -= 16;
            d += 16;
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
lasx_fill_u32 (uint32_t *bits,
               int       stride,
               int       x,
               int       y,
               int       width,
               int       height,
               uint32_t  filler)
{
    __m256i xvfill = __lasx_xvreplgr2vr_w(filler);
    bits += y * stride + x;

    while (height--) {
        int w = width;
        uint32_t *d = bits;

        while (w && ((uintptr_t)d & 31)) {
            *d = filler;
            w--;
            d++;
        }

        while (w >= 32) {
            __lasx_xvst(xvfill, d, 0);
            __lasx_xvst(xvfill, d, 32);
            __lasx_xvst(xvfill, d, 64);
            __lasx_xvst(xvfill, d, 96);
            w -= 32;
            d += 32;
        }

        if (w >= 16) {
            __lasx_xvst(xvfill, d, 0);
            __lasx_xvst(xvfill, d, 32);
            w -= 16;
            d += 16;
        }

        if (w >= 8) {
            __lasx_xvst(xvfill, d, 0);
            w -= 8;
            d += 8;
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
lasx_fill (pixman_implementation_t *imp,
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
            lasx_fill_u8 ((uint8_t *)bits, stride, x, y, width, height, (uint8_t)filler);
            return TRUE;

        case 16:
            lasx_fill_u16 ((uint16_t *)bits, stride, x, y, width, height, (uint16_t)filler);
            return TRUE;

        case 32:
            lasx_fill_u32 (bits, stride, x, y, width, height, filler);
            return TRUE;

        default:
            return FALSE;
    }

    return TRUE;
}


static void
lasx_composite_over_n_8_8888 (pixman_implementation_t *imp,
                              pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t src, srca;
    uint32_t *dst_line, *dst, d;
    uint8_t  *mask_line, *mask, m;
    int dst_stride, mask_stride;
    int32_t w;
    v8u32 vsrca, vsrc;
    __m256i vff;

    src   = _pixman_image_get_solid(imp, src_image, dest_image->bits.format);
    vsrc  = (v8u32)__lasx_xvreplgr2vr_w(src);
    srca  = src >> 24;
    vsrca = (v8u32)__lasx_xvreplgr2vr_w(srca);
    vff   = __lasx_xvreplgr2vr_w(0xff);

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

        while (w >= 8) {
            v8u32 ma = {mask[0], mask[1], mask[2], mask[3],
                        mask[4], mask[5], mask[6], mask[7]};

            if (__lasx_xbnz_w(__lasx_xvseqi_w((__m256i)ma, 0xff))){
                if (__lasx_xbnz_w(__lasx_xvseqi_w(vsrca, 0xff)))
                    *(__m256i*) dst = (__m256i)vsrc;
                else if (__lasx_xbnz_w(__lasx_xvsub_w((__m256i)ma, vff)))
                    *(__m256i*) dst = lasx_over_u((__m256i)vsrc, *(__m256i*)dst);
            } else if (__lasx_xbnz_w((__m256i)ma)) {
                __m256i d0 = lasx_in_u((__m256i)vsrc, (__m256i)ma);
                *(__m256i*) dst = lasx_over_u(d0, *(__m256i*)dst);
            } else {
                for (int i = 0; i < 8; i++) {
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
            dst += 8;
            w -= 8;
            mask += 8;
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
lasx_composite_add_8_8 (pixman_implementation_t *imp,
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

        lasx_combine_add_u(imp, op, (uint32_t *)dst, (uint32_t *)src, NULL, w >> 2);
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
lasx_composite_add_8888_8888 (pixman_implementation_t *imp,
                              pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t *dst_line;
    uint32_t *src_line;
    int dst_stride, src_stride;

    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint32_t, src_stride, src_line, 1);
    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);

    while (height--) {
        lasx_combine_add_u(imp, op, dst_line, src_line, NULL, width);
        dst_line += dst_stride;
        src_line += src_stride;
    }
}

static void
lasx_composite_over_8888_8888 (pixman_implementation_t *imp,
                               pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    int dst_stride, src_stride;
    uint32_t *dst_line;
    uint32_t *src_line;

    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint32_t, src_stride, src_line, 1);
    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);

    while (height--) {
        lasx_combine_over_u_no_mask (dst_line, src_line, width);
        dst_line += dst_stride;
        src_line += src_stride;
    }
}

static void
lasx_composite_copy_area (pixman_implementation_t *imp,
                          pixman_composite_info_t *info)
{
    bits_image_t src_bits, dst_bits;
    src_bits = info->src_image->bits;
    dst_bits = info->dest_image->bits;
    lasx_blt (imp, src_bits.bits,
              dst_bits.bits,
              src_bits.rowstride,
              dst_bits.rowstride,
              PIXMAN_FORMAT_BPP (src_bits.format),
              PIXMAN_FORMAT_BPP (dst_bits.format),
              info->src_x, info->src_y, info->dest_x,
              info->dest_y, info->width, info->height);
}

static void
lasx_composite_src_x888_0565 (pixman_implementation_t *imp,
                              pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint16_t *dst_line, *dst;
    uint32_t *src_line, *src, s;
    int dst_stride, src_stride;
    int32_t w;

    __m256i src0, src1;
    __m256i rb0, rb1, t0, t1, g0, g1;
    __m256i tmp;
    __m256i mask_565_rb = __lasx_xvreplgr2vr_w(0x00f800f8);
    __m256i mask_multiplier = __lasx_xvreplgr2vr_w(0x20000004);
    __m256i mask_green_4x32 = __lasx_xvreplgr2vr_w(0x0000fc00);

    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint32_t, src_stride, src_line, 1);
    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint16_t, dst_stride, dst_line, 1);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        src = src_line;
        src_line += src_stride;
        w = width;

        while (w && (uintptr_t)src & 31) {
            s = *src++;
            *dst = convert_8888_to_0565(s);
            dst++;
            w--;
        }

        while (w >= 16) {
            src0 = __lasx_xvld(src, 0);
            src1 = __lasx_xvld(src, 32);
            src += 16;
            w -= 16;

            rb0 = src0 & mask_565_rb;
            rb1 = src1 & mask_565_rb;
            t0 = __lasx_xvdp2_w_h(rb0, mask_multiplier);
            t1 = __lasx_xvdp2_w_h(rb1, mask_multiplier);
            g0 = src0 & mask_green_4x32;
            g1 = src1 & mask_green_4x32;
            t0 |= g0;
            t1 |= g1;
            t0 = __lasx_xvslli_w(t0, 11);
            t1 = __lasx_xvslli_w(t1, 11);
            t0 = __lasx_xvsrai_w(t0, 16);
            t1 = __lasx_xvsrai_w(t1, 16);
            t0 = __lasx_xvsat_h(t0, 15);
            t1 = __lasx_xvsat_h(t1, 15);
            tmp = __lasx_xvpickev_h(t1, t0);
            tmp = __lasx_xvpermi_d(tmp, 0xd8);
            __lasx_xvst(tmp, dst, 0);
            dst += 16;
        }

        if (w >= 8) {
            src0 = __lasx_xvld(src, 0);
            src += 8;
            w -= 8;

            rb0 = src0 & mask_565_rb;
            t0 = __lasx_xvdp2_w_h(rb0, mask_multiplier);
            g0 = src0 & mask_green_4x32;
            t0 |= g0;
            t0 = __lasx_xvslli_w(t0, 11);
            t0 = __lasx_xvsrai_w(t0, 16);
            t0 = __lasx_xvsat_h(t0, 15);
            tmp = __lasx_xvpickev_h(t0, t0);
            __lasx_xvstelm_d(tmp, dst, 0, 0);
            __lasx_xvstelm_d(tmp, dst, 8, 2);
            dst += 8;
        }

        while (w--) {
            s = *src++;
            *dst = convert_8888_to_0565(s);
            dst++;
        }
    }
}

static void
lasx_composite_in_n_8_8 (pixman_implementation_t *imp,
                        pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS(info);
    uint8_t *dst_line, *dst;
    uint8_t *mask_line, *mask;
    int dst_stride, mask_stride;
    uint32_t m, src, srca;
    int32_t w;
    uint16_t t;

    __m256i alpha, tmp;
    __m256i vmask, vmask_lo, vmask_hi;
    __m256i vdst, vdst_lo, vdst_hi;
    __m256i mask_zero = __lasx_xvldi(0);

    PIXMAN_IMAGE_GET_LINE(dest_image, dest_x, dest_y, uint8_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE(mask_image, mask_x, mask_y, uint8_t, mask_stride, mask_line, 1);

    src = _pixman_image_get_solid(imp, src_image, dest_image->bits.format);
    srca = src >> 24;
    alpha = __lasx_xvreplgr2vr_w(src);
    alpha = __lasx_xvilvl_b(mask_zero, alpha);
    alpha = __lasx_xvshuf4i_w(alpha, 0x44);
    alpha = __lasx_xvshuf4i_h(alpha, 0xff);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        mask = mask_line;
        mask_line += mask_stride;
        w = width;

        while (w >= 32) {
            vmask = __lasx_xvld(mask, 0);
            vdst = __lasx_xvld(dst, 0);
            mask += 32;
            w -= 32;

            vmask_lo = __lasx_vext2xv_hu_bu(vmask);
            vdst_lo = __lasx_vext2xv_hu_bu(vdst);
            vmask_hi = __lasx_xvpermi_q(vmask, vmask, 0x03);
            vdst_hi = __lasx_xvpermi_q(vdst, vdst, 0x03);
            vmask_hi = __lasx_vext2xv_hu_bu(vmask_hi);
            vdst_hi = __lasx_vext2xv_hu_bu(vdst_hi);
            vmask_lo = lasx_pix_multiply(alpha, vmask_lo);
            vmask_hi = lasx_pix_multiply(alpha, vmask_hi);
            vdst_lo = lasx_pix_multiply(vmask_lo, vdst_lo);
            vdst_hi = lasx_pix_multiply(vmask_hi, vdst_hi);
            vdst_lo = __lasx_xvsat_bu(vdst_lo, 7);
            vdst_hi = __lasx_xvsat_bu(vdst_hi, 7);
            tmp = __lasx_xvpickev_b(vdst_hi, vdst_lo);
            tmp = __lasx_xvpermi_d(tmp, 0xd8);
            __lasx_xvst(tmp, dst, 0);
            dst += 32;
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
lasx_composite_in_8_8 (pixman_implementation_t *imp,
                       pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint8_t *dst_line, *dst;
    uint8_t *src_line, *src;
    int src_stride, dst_stride;
    int32_t w, s;
    uint16_t t;

    __m256i tmp;
    __m256i vsrc, vsrc_lo, vsrc_hi;
    __m256i vdst, vdst_lo, vdst_hi;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint8_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint8_t, src_stride, src_line, 1);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        src = src_line;
        src_line += src_stride;
        w = width;

        while (w >= 32) {
            vsrc = __lasx_xvld(src, 0);
            vdst = __lasx_xvld(dst, 0);
            src += 32;
            w -= 32;

            vsrc_lo = __lasx_vext2xv_hu_bu(vsrc);
            vdst_lo = __lasx_vext2xv_hu_bu(vdst);
            vsrc_hi = __lasx_xvpermi_q(vsrc, vsrc, 0x03);
            vdst_hi = __lasx_xvpermi_q(vdst, vdst, 0x03);
            vsrc_hi = __lasx_vext2xv_hu_bu(vsrc_hi);
            vdst_hi = __lasx_vext2xv_hu_bu(vdst_hi);
            vdst_lo = lasx_pix_multiply(vsrc_lo, vdst_lo);
            vdst_hi = lasx_pix_multiply(vsrc_hi, vdst_hi);
            vdst_lo = __lasx_xvsat_bu(vdst_lo, 7);
            vdst_hi = __lasx_xvsat_bu(vdst_hi, 7);
            tmp = __lasx_xvpickev_b(vdst_hi, vdst_lo);
            tmp = __lasx_xvpermi_d(tmp, 0xd8);
            __lasx_xvst(tmp, dst, 0);
            dst += 32;
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
lasx_composite_over_n_8888_8888_ca (pixman_implementation_t *imp,
                                    pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t src, srca, ns;
    uint32_t *dst_line, *dst, nd;
    uint32_t *mask_line, *mask, ma;
    int dst_stride, mask_stride;
    int32_t w;

    __m256i d, m, t;
    __m256i tmp0, tmp1;
    __m256i s, sa, d0, d1, m0, m1, t0, t1;
    __m256i zero = __lasx_xvldi(0);
    __m256i bit_set = __lasx_xvreplgr2vr_h(0xff);
    src = _pixman_image_get_solid (imp, src_image, dest_image->bits.format);
    srca = src >> 24;
    if (src == 0)
        return;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (mask_image, mask_x, mask_y, uint32_t, mask_stride, mask_line, 1);
    s = __lasx_xvreplgr2vr_w(src);
    tmp0 = __lasx_xvilvl_b(zero, s);
    tmp1 = __lasx_xvilvh_b(zero, s);
    s = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
    sa = __lasx_xvshuf4i_h(s, 0xff);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        mask = mask_line;
        mask_line += mask_stride;
        w = width;

        while (w && ((uintptr_t)dst & 31)) {
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

        while (w >= 8) {
            m = __lasx_xvld(mask, 0);
            mask += 8;
            w -= 8;

            if (__lasx_xbnz_v(m)) {
                d = __lasx_xvld(dst, 0);
                d0 = __lasx_vext2xv_hu_bu(d);
                m0 = __lasx_vext2xv_hu_bu(m);
                d1 = __lasx_xvpermi_q(d, d, 0x03);
                m1 = __lasx_xvpermi_q(m, m, 0x03);
                d1 = __lasx_vext2xv_hu_bu(d1);
                m1 = __lasx_vext2xv_hu_bu(m1);

                t0 = lasx_pix_multiply(s, m0);
                t1 = lasx_pix_multiply(s, m1);

                m0 = lasx_pix_multiply(m0, sa);
                m1 = lasx_pix_multiply(m1, sa);
                m0 = __lasx_xvxor_v(m0, bit_set);
                m1 = __lasx_xvxor_v(m1, bit_set);
                d0 = lasx_pix_multiply(d0, m0);
                d1 = lasx_pix_multiply(d1, m1);

                d = __lasx_xvpickev_b(d1, d0);
                t = __lasx_xvpickev_b(t1, t0);
                d = __lasx_xvpermi_d(d, 0xd8);
                t = __lasx_xvpermi_d(t, 0xd8);
                d = __lasx_xvsadd_bu(d, t);
                __lasx_xvst(d, dst, 0);
            }
            dst += 8;
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
lasx_composite_over_reverse_n_8888 (pixman_implementation_t *imp,
                                    pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t src;
    uint32_t *dst_line, *dst;
    int dst_stride;
    int32_t w;

    __m256i d, t;
    __m256i s, d0, d1;
    __m256i tmp0, tmp1;
    __m256i zero = __lasx_xvldi(0);
    __m256i bit_set = __lasx_xvreplgr2vr_h(0xff);
    src = _pixman_image_get_solid (imp, src_image, dest_image->bits.format);
    if (src == 0)
        return;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);
    s = __lasx_xvreplgr2vr_w(src);
    tmp0 = __lasx_xvilvl_b(zero, s);
    tmp1 = __lasx_xvilvh_b(zero, s);
    s = __lasx_xvpermi_q(tmp0, tmp1, 0x02);

    while (height--)
    {
        dst = dst_line;
        dst_line += dst_stride;
        w = width;

        while (w && ((uintptr_t)dst & 31)) {
            d = __lasx_xvldrepl_w(dst, 0);
            d0 = __lasx_vext2xv_hu_bu(d);
            d0 = __lasx_xvshuf4i_h(d0, 0xff);
            d0 = __lasx_xvxor_v(d0, bit_set);
            d0 = lasx_pix_multiply(d0, s);
            t = __lasx_xvpickev_b(zero, d0);
            t = __lasx_xvpermi_d(t, 0xd8);
            d = __lasx_xvsadd_bu(d, t);
            __lasx_xvstelm_w(d, dst, 0, 0);
            dst += 1;
            w--;
        }

        while (w >= 8) {
            d = __lasx_xvld(dst, 0);
            w -= 8;

            d0 = __lasx_vext2xv_hu_bu(d);
            d1 = __lasx_xvpermi_q(d, d, 0x03);
            d1 = __lasx_vext2xv_hu_bu(d1);
            d0 = __lasx_xvshuf4i_h(d0, 0xff);
            d1 = __lasx_xvshuf4i_h(d1, 0xff);
            d0 = __lasx_xvxor_v(d0, bit_set);
            d1 = __lasx_xvxor_v(d1, bit_set);
            d0 = lasx_pix_multiply(d0, s);
            d1 = lasx_pix_multiply(d1, s);
            t = __lasx_xvpickev_b(d1, d0);
            t = __lasx_xvpermi_d(t, 0xd8);
            d = __lasx_xvsadd_bu(d, t);
            __lasx_xvst(d, dst, 0);
            dst += 8;
        }

        while (w--) {
            d = __lasx_xvldrepl_w(dst, 0);
            d0 = __lasx_vext2xv_hu_bu(d);
            d0 = __lasx_xvshuf4i_h(d0, 0xff);
            d0 = __lasx_xvxor_v(d0, bit_set);
            d0 = lasx_pix_multiply(d0, s);
            t = __lasx_xvpickev_b(zero, d0);
            t = __lasx_xvpermi_d(t, 0xd8);
            d = __lasx_xvsadd_bu(d, t);
            __lasx_xvstelm_w(d, dst, 0, 0);
            dst += 1;
        }
    }
}

static void
lasx_composite_src_x888_8888 (pixman_implementation_t *imp,
                              pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t *dst_line, *dst;
    uint32_t *src_line, *src;
    int32_t w;
    int dst_stride, src_stride;
    __m256i mask = mask_ff000000;
    __m256i vsrc0, vsrc1, vsrc2, vsrc3, vsrc4, vsrc5, vsrc6, vsrc7;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint32_t, src_stride, src_line, 1);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        src = src_line;
        src_line += src_stride;
        w = width;

        while (w && ((uintptr_t)dst & 31)) {
            *dst++ = *src++ | 0xff000000;
            w--;
        }

        while (w >= 64) {
            vsrc0 = __lasx_xvld(src, 0);
            vsrc1 = __lasx_xvld(src, 32);
            vsrc2 = __lasx_xvld(src, 64);
            vsrc3 = __lasx_xvld(src, 96);
            vsrc4 = __lasx_xvld(src, 128);
            vsrc5 = __lasx_xvld(src, 160);
            vsrc6 = __lasx_xvld(src, 192);
            vsrc7 = __lasx_xvld(src, 224);
            vsrc0 = __lasx_xvor_v(vsrc0, mask);
            vsrc1 = __lasx_xvor_v(vsrc1, mask);
            vsrc2 = __lasx_xvor_v(vsrc2, mask);
            vsrc3 = __lasx_xvor_v(vsrc3, mask);
            vsrc4 = __lasx_xvor_v(vsrc4, mask);
            vsrc5 = __lasx_xvor_v(vsrc5, mask);
            vsrc6 = __lasx_xvor_v(vsrc6, mask);
            vsrc7 = __lasx_xvor_v(vsrc7, mask);
            __lasx_xvst(vsrc0, dst, 0);
            __lasx_xvst(vsrc1, dst, 32);
            __lasx_xvst(vsrc2, dst, 64);
            __lasx_xvst(vsrc3, dst, 96);
            __lasx_xvst(vsrc4, dst, 128);
            __lasx_xvst(vsrc5, dst, 160);
            __lasx_xvst(vsrc6, dst, 192);
            __lasx_xvst(vsrc7, dst, 224);

            src += 64;
            w -= 64;
            dst += 64;
        }

        if (w >= 32) {
            vsrc0 = __lasx_xvld(src, 0);
            vsrc1 = __lasx_xvld(src, 32);
            vsrc2 = __lasx_xvld(src, 64);
            vsrc3 = __lasx_xvld(src, 96);
            vsrc0 = __lasx_xvor_v(vsrc0, mask);
            vsrc1 = __lasx_xvor_v(vsrc1, mask);
            vsrc2 = __lasx_xvor_v(vsrc2, mask);
            vsrc3 = __lasx_xvor_v(vsrc3, mask);
            __lasx_xvst(vsrc0, dst, 0);
            __lasx_xvst(vsrc1, dst, 32);
            __lasx_xvst(vsrc2, dst, 64);
            __lasx_xvst(vsrc3, dst, 96);

            src += 32;
            w -= 32;
            dst += 32;
        }

        if (w >= 16) {
            vsrc0 = __lasx_xvld(src, 0);
            vsrc1 = __lasx_xvld(src, 32);
            vsrc0 = __lasx_xvor_v(vsrc0, mask);
            vsrc1 = __lasx_xvor_v(vsrc1, mask);
            __lasx_xvst(vsrc0, dst, 0);
            __lasx_xvst(vsrc1, dst, 32);

            src += 16;
            w -= 16;
            dst += 16;
        }

        if (w >= 8) {
            vsrc0 = __lasx_xvld(src, 0);
            vsrc0 = __lasx_xvor_v(vsrc0, mask);
            __lasx_xvst(vsrc0, dst, 0);

            src += 8;
            w -= 8;
            dst += 8;
        }

        while (w--) {
            *dst++ = *src++ | 0xff000000;
        }
    }
}

static void
lasx_composite_add_n_8_8 (pixman_implementation_t *imp,
                          pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint8_t *dst_line, *dst;
    uint8_t *mask_line, *mask;
    int dst_stride, mask_stride;
    int32_t w;
    uint32_t src;
    uint16_t sa;

    __m256i d0;
    __m256i vsrc, t0, t1;
    __m256i a0, a0_l, a0_h;
    __m256i b0, b0_l, b0_h;
    __m256i zero = __lasx_xvldi(0);
    __m256i one_half = __lasx_xvreplgr2vr_h(0x80);
    __m256i g_shift  = __lasx_xvreplgr2vr_h(8);

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint8_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (mask_image, mask_x, mask_y, uint8_t, mask_stride, mask_line, 1);

    src = _pixman_image_get_solid (imp, src_image, dest_image->bits.format);

    sa = (src >> 24);
    vsrc = __lasx_xvreplgr2vr_h(sa);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        mask = mask_line;
        mask_line += mask_stride;
        w = width;

        while (w >= 32) {
            a0 = __lasx_xvld(mask, 0);
            w -= 32;
            mask += 32;

            a0_l = __lasx_vext2xv_hu_bu(a0);
            a0_h = __lasx_xvpermi_q(a0, a0, 0x03);
            a0_h = __lasx_vext2xv_hu_bu(a0_h);

            a0_l = __lasx_xvmadd_h(one_half, a0_l, vsrc);
            a0_h = __lasx_xvmadd_h(one_half, a0_h, vsrc);

            a0_l = __lasx_xvsadd_hu(__lasx_xvsrl_h(a0_l, g_shift), a0_l);
            a0_h = __lasx_xvsadd_hu(__lasx_xvsrl_h(a0_h, g_shift), a0_h);

            a0_l = __lasx_xvsrl_h(a0_l, g_shift);
            a0_h = __lasx_xvsrl_h(a0_h, g_shift);

            b0 = __lasx_xvld(dst, 0);
            b0_l = __lasx_vext2xv_hu_bu(b0);
            b0_h = __lasx_xvpermi_q(b0, b0, 0x03);
            b0_h = __lasx_vext2xv_hu_bu(b0_h);

            t0 = __lasx_xvadd_h(a0_l, b0_l);
            t1 = __lasx_xvadd_h(a0_h, b0_h);

            t0 = __lasx_xvor_v(t0, __lasx_xvsub_h(zero, __lasx_xvsrl_h(t0, g_shift)));
            t1 = __lasx_xvor_v(t1, __lasx_xvsub_h(zero, __lasx_xvsrl_h(t1, g_shift)));

            t0 = __lasx_xvsat_hu(t0, 7);
            t1 = __lasx_xvsat_hu(t1 ,7);

            d0 = __lasx_xvpickev_b(t1, t0);
            d0 = __lasx_xvpermi_d(d0, 0xd8);
            __lasx_xvst(d0, dst, 0);
            dst += 32;
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
lasx_composite_add_n_8 (pixman_implementation_t *imp,
                        pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint8_t *dst_line, *dst;
    int dst_stride;
    int32_t w;
    uint32_t src;

    __m256i vsrc, d0, d1;

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

    vsrc = __lasx_xvreplgr2vr_b(src);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        w = width;

        while (w && ((uintptr_t)dst & 31)) {
            d0 = __lasx_xvldrepl_b(dst, 0);
            d0 = __lasx_xvsadd_bu(vsrc, d0);
            __lasx_xvstelm_b(d0, dst, 0, 0);
            dst++;
            w--;
        }

        while (w >= 64) {
            d0 = __lasx_xvld(dst, 0);
            d1 = __lasx_xvld(dst, 32);
            w -= 64;
            d0 = __lasx_xvsadd_bu(vsrc, d0);
            d1 = __lasx_xvsadd_bu(vsrc, d1);
            __lasx_xvst(d0, dst, 0);
            __lasx_xvst(d1, dst, 32);
            dst += 64;
        }

        if (w >= 32) {
            d0 = __lasx_xvld(dst, 0);
            w -= 32;
            d0 = __lasx_xvsadd_bu(vsrc, d0);
            __lasx_xvst(d0, dst, 0);
            dst += 32;
        }

        if (w >= 8) {
            d0 = __lasx_xvldrepl_d(dst, 0);
            w -= 8;
            d0 = __lasx_xvsadd_bu(vsrc, d0);
            __lasx_xvstelm_d(d0, dst, 0, 0);
            dst += 8;
        }

        if (w >= 4) {
            d0 = __lasx_xvldrepl_w(dst, 0);
            w -= 4;
            d0 = __lasx_xvsadd_bu(vsrc, d0);
            __lasx_xvstelm_w(d0, dst, 0, 0);
            dst += 4;
        }

        while (w--) {
            d0 = __lasx_xvldrepl_b(dst, 0);
            d0 = __lasx_xvsadd_bu(vsrc, d0);
            __lasx_xvstelm_b(d0, dst, 0, 0);
            dst++;
        }
    }
}

static void
lasx_composite_add_n_8888 (pixman_implementation_t *imp,
                           pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t *dst_line, *dst, src;
    int dst_stride, w;

    __m256i vsrc, d0, d1;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);

    src = _pixman_image_get_solid (imp, src_image, dest_image->bits.format);
    if (src == 0)
        return;

    if (src == ~0) {
        pixman_fill (dest_image->bits.bits, dest_image->bits.rowstride, 32,
                     dest_x, dest_y, width, height, ~0);
        return;
    }

    vsrc = __lasx_xvreplgr2vr_w(src);

    while (height--) {
        w = width;

        dst = dst_line;
        dst_line += dst_stride;

        while (w && (uintptr_t)dst & 31) {
            d0 = __lasx_xvldrepl_w(dst, 0);
            d0 = __lasx_xvsadd_bu(vsrc, d0);
            __lasx_xvstelm_w(d0, dst, 0, 0);
            dst++;
            w--;
        }

        while (w >= 16) {
            d0 = __lasx_xvld(dst, 0);
            d1 = __lasx_xvld(dst, 32);
            w -= 16;
            d0 = __lasx_xvsadd_bu(vsrc, d0);
            d1 = __lasx_xvsadd_bu(vsrc, d1);
            __lasx_xvst(d0, dst, 0);
            __lasx_xvst(d1, dst, 32);
            dst += 16;
        }

        if (w >= 8) {
            d0 = __lasx_xvld(dst, 0);
            w -= 8;
            d0 = __lasx_xvsadd_bu(vsrc, d0);
            __lasx_xvst(d0, dst, 0);
            dst += 8;
        }

        while (w--) {
            d0 = __lasx_xvldrepl_w(dst, 0);
            d0 = __lasx_xvsadd_bu(vsrc, d0);
            __lasx_xvstelm_w(d0, dst, 0, 0);
            dst++;
        }
    }
}

static force_inline __m256i
unpack_32_1x256 (uint32_t data)
{
    __m256i zero = __lasx_xvldi(0);
    __m256i tmp = __lasx_xvinsgr2vr_w(zero, data, 0);
    return __lasx_xvilvl_b(zero, tmp);
}

static force_inline __m256i
unpack_32_2x256 (uint32_t data)
{
    __m256i tmp0, out0;
    __m256i zero = __lasx_xvldi(0);
    tmp0 = __lasx_xvinsgr2vr_w(tmp0, data, 0);
    tmp0 = __lasx_xvpermi_q(tmp0, tmp0, 0x20);
    out0 = __lasx_xvilvl_b(zero, tmp0);

    return out0;
}

static force_inline __m256i
expand_pixel_32_1x256 (uint32_t data)
{
    return __lasx_xvshuf4i_w(unpack_32_1x256(data), 0x44);
}

static force_inline __m256i
expand_pixel_32_2x256 (uint32_t data)
{
    return __lasx_xvshuf4i_w(unpack_32_2x256(data), 0x44);
}

static force_inline __m256i
expand_alpha_1x256 (__m256i data)
{
    return __lasx_xvshuf4i_h(data, 0xff);
}

static force_inline __m256i
expand_alphaa_2x256 (__m256i data)
{
    __m256i tmp0;
    tmp0 = __lasx_xvshuf4i_h(data, 0xff);
    tmp0 = __lasx_xvpermi_q(tmp0, tmp0, 0x20);

    return tmp0;
}

static force_inline __m256i
unpack_565_to_8888 (__m256i lo)
{
    __m256i r, g, b, rb, t;
    __m256i mask_green_4x32 = __lasx_xvreplgr2vr_w(0x0000fc00);
    __m256i mask_red_4x32   = __lasx_xvreplgr2vr_w(0x00f80000);
    __m256i mask_blue_4x32  = __lasx_xvreplgr2vr_w(0x000000f8);
    __m256i mask_565_fix_rb = __lasx_xvreplgr2vr_w(0x00e000e0);
    __m256i mask_565_fix_g  = __lasx_xvreplgr2vr_w(0x0000c000);

    r  = __lasx_xvslli_w(lo, 8);
    r  = __lasx_xvand_v(r, mask_red_4x32);
    g  = __lasx_xvslli_w(lo, 5);
    g  = __lasx_xvand_v(g, mask_green_4x32);
    b  = __lasx_xvslli_w(lo, 3);
    b  = __lasx_xvand_v(b, mask_blue_4x32);

    rb = __lasx_xvor_v(r, b);
    t  = __lasx_xvand_v(rb, mask_565_fix_rb);
    t  = __lasx_xvsrli_w(t, 5);
    rb = __lasx_xvor_v(rb, t);

    t  = __lasx_xvand_v(g, mask_565_fix_g);
    t  = __lasx_xvsrli_w(t, 6);
    g  = __lasx_xvor_v(g, t);

    return (__lasx_xvor_v(rb, g));
}

static force_inline void
unpack_256_2x256 (__m256i data, __m256i *data_lo, __m256i *data_hi)
{
    __m256i mask_zero = __lasx_xvldi(0);
    *data_lo = __lasx_xvilvl_b(mask_zero, data);
    *data_hi = __lasx_xvilvh_b(mask_zero, data);
}

static force_inline void
unpack_565_256_4x256 (__m256i data, __m256i *data0,
                      __m256i *data1, __m256i *data2, __m256i *data3)
{
    __m256i lo, hi;
    __m256i zero = __lasx_xvldi(0);
    lo = __lasx_xvilvl_h(zero, data);
    hi = __lasx_xvilvh_h(zero, data);
    lo = unpack_565_to_8888(lo);
    hi = unpack_565_to_8888(hi);

    unpack_256_2x256((__m256i)lo, (__m256i*)data0, (__m256i*)data1);
    unpack_256_2x256((__m256i)hi, (__m256i*)data2, (__m256i*)data3);
}

static force_inline void
negate_2x256 (__m256i data_lo, __m256i data_hi, __m256i *neg_lo, __m256i *neg_hi)
{
    *neg_lo = __lasx_xvxor_v(data_lo, mask_00ff);
    *neg_hi = __lasx_xvxor_v(data_hi, mask_00ff);
}

static force_inline void
over_2x256 (__m256i *src_lo, __m256i *src_hi, __m256i *alpha_lo,
            __m256i *alpha_hi, __m256i *dst_lo, __m256i *dst_hi)
{
    __m256i t1, t2;
    negate_2x256(*alpha_lo, *alpha_hi, &t1, &t2);
    *dst_lo = lasx_pix_multiply(*dst_lo, t1);
    *dst_hi = lasx_pix_multiply(*dst_hi, t2);
    *dst_lo = __lasx_xvsadd_bu(*src_lo, *dst_lo);
    *dst_hi = __lasx_xvsadd_bu(*src_hi, *dst_hi);
}

static force_inline __m256i
pack_2x256_256 (__m256i lo, __m256i hi)
{
    __m256i tmp0 = __lasx_xvsat_bu(lo, 7);
    __m256i tmp1 = __lasx_xvsat_bu(hi, 7);
    __m256i tmp2 = __lasx_xvpickev_b(tmp1, tmp0);

    return tmp2;
}

static force_inline __m256i
pack_565_2x256_256 (__m256i lo, __m256i hi)
{
    __m256i data;
    __m256i r, g1, g2, b;
    __m256i mask_565_r  = __lasx_xvreplgr2vr_w(0x00f80000);
    __m256i mask_565_g1 = __lasx_xvreplgr2vr_w(0x00070000);
    __m256i mask_565_g2 = __lasx_xvreplgr2vr_w(0x000000e0);
    __m256i mask_565_b  = __lasx_xvreplgr2vr_w(0x0000001f);

    data = pack_2x256_256 (lo, hi);
    r    = __lasx_xvand_v(data, mask_565_r);
    g1   = __lasx_xvslli_w(data, 3) & mask_565_g1;
    g2   = __lasx_xvsrli_w(data, 5) & mask_565_g2;
    b    = __lasx_xvsrli_w(data, 3) & mask_565_b;

    return (((r|g1)|g2)|b);
}

static force_inline __m256i
expand565_16_1x256 (uint16_t pixel)
{
    __m256i m;
    __m256i zero = __lasx_xvldi(0);

    m = __lasx_xvinsgr2vr_w(m, pixel, 0);
    m = unpack_565_to_8888(m);
    m = __lasx_xvilvl_b(zero, m);

    return m;
}

static force_inline uint32_t
pack_1x256_32 (__m256i data)
{
    __m256i tmp0, tmp1;
    __m256i zero = __lasx_xvldi(0);

    tmp0 = __lasx_xvsat_bu(data, 7);
    tmp1 = __lasx_xvpickev_b(zero, tmp0);

    return (__lasx_xvpickve2gr_wu(tmp1, 0));
}

static force_inline uint16_t
pack_565_32_16 (uint32_t pixel)
{
    return (uint16_t)(((pixel >> 8) & 0xf800) |
                      ((pixel >> 5) & 0x07e0) |
                      ((pixel >> 3) & 0x001f));
}

static force_inline __m256i
pack_565_4x256_256 (__m256i *v0, __m256i *v1, __m256i *v2, __m256i *v3)
{
    return pack_2x256_256(pack_565_2x256_256(*v0, *v1),
                          pack_565_2x256_256(*v2, *v3));
}

static force_inline void
expand_alpha_2x256 (__m256i data_lo, __m256i data_hi, __m256i *alpha_lo, __m256i *alpha_hi)
{
    *alpha_lo = __lasx_xvshuf4i_h(data_lo, 0xff);
    *alpha_hi = __lasx_xvshuf4i_h(data_hi, 0xff);
}

static force_inline void
expand_alpha_rev_2x256 (__m256i data_lo,  __m256i data_hi, __m256i *alpha_lo, __m256i *alpha_hi)
{
    *alpha_lo = __lasx_xvshuf4i_h(data_lo, 0x00);
    *alpha_hi = __lasx_xvshuf4i_h(data_hi, 0x00);
}

static force_inline uint16_t
composite_over_8888_0565pixel (uint32_t src, uint16_t dst)
{
    __m256i ms;
    ms = unpack_32_1x256(src);

    return pack_565_32_16(pack_1x256_32((__m256i)over_1x256((__m256i)ms,
                          (__m256i)expand_alpha_1x256((__m256i)ms), expand565_16_1x256(dst))));
}

static force_inline void
in_over_2x256 (__m256i *src_lo, __m256i *src_hi, __m256i *alpha_lo, __m256i *alpha_hi,
               __m256i *mask_lo, __m256i *mask_hi, __m256i *dst_lo, __m256i *dst_hi)
{
    __m256i s_lo, s_hi;
    __m256i a_lo, a_hi;
    s_lo = lasx_pix_multiply(*src_lo, *mask_lo);
    s_hi = lasx_pix_multiply(*src_hi, *mask_hi);
    a_lo = lasx_pix_multiply(*alpha_lo, *mask_lo);
    a_hi = lasx_pix_multiply(*alpha_hi, *mask_hi);
    over_2x256(&s_lo, &s_hi, &a_lo, &a_hi, dst_lo, dst_hi);
}

static force_inline __m256i
in_over_1x256 (__m256i *src, __m256i *alpha, __m256i *mask, __m256i *dst)
{
    return over_1x256(lasx_pix_multiply(*src, *mask),
                      lasx_pix_multiply(*alpha, *mask), *dst);
}

static force_inline __m256i
expand_alpha_rev_1x256 (__m256i data)
{
    __m256i v0 = {0x00000000, 0x00000000, 0xffffffff, 0xffffffff};
    __m256i v_hi = __lasx_xvand_v(data, v0);
    data = __lasx_xvshuf4i_h(data, 0x00);
    v0 = __lasx_xvnor_v(v0, v0);
    data = __lasx_xvand_v(data, v0);
    data = __lasx_xvor_v(data, v_hi);

    return data;
}

static void
lasx_composite_over_n_0565 (pixman_implementation_t *imp,
                            pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t src;
    uint16_t *dst_line, *dst, d;
    int32_t w;
    int dst_stride;
    __m256i vsrc, valpha;
    __m256i vdst, vdst0, vdst1, vdst2, vdst3;

    src = _pixman_image_get_solid (imp, src_image, dest_image->bits.format);

    if (src == 0)
        return;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint16_t, dst_stride, dst_line, 1);

    vsrc = expand_pixel_32_1x256(src);
    valpha = expand_alpha_1x256(vsrc);

    while (height--) {
        dst = dst_line;

        dst_line += dst_stride;
        w = width;

        while (w >= 16) {
            vdst = __lasx_xvld(dst, 0);
            w -= 16;
            vsrc = __lasx_xvpermi_q(vsrc, vsrc, 0x20);
            valpha = __lasx_xvpermi_q(valpha, valpha, 0x20);

            unpack_565_256_4x256(vdst, &vdst0, &vdst1, &vdst2, &vdst3);

            over_2x256(&vsrc, &vsrc, &valpha, &valpha, &vdst0, &vdst1);
            over_2x256(&vsrc, &vsrc, &valpha, &valpha, &vdst2, &vdst3);

            vdst = pack_565_4x256_256(&vdst0, &vdst1, &vdst2, &vdst3);
            __lasx_xvst(vdst, dst, 0);
            dst += 16;
        }

        while (w--) {
            d = *dst;
            *dst++ = pack_565_32_16(pack_1x256_32(
                                    (over_1x256(vsrc,valpha, expand565_16_1x256(d)))));
        }
    }
}

static void
lasx_composite_over_8888_0565 (pixman_implementation_t *imp,
                               pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint16_t *dst_line, *dst, d;
    uint32_t *src_line, *src, s;
    int dst_stride, src_stride;
    int32_t w;

    __m256i valpha_lo, valpha_hi;
    __m256i vsrc, vsrc_lo, vsrc_hi;
    __m256i vdst, vdst0, vdst1, vdst2, vdst3;
    __m256i dst0, dst1, dst2, dst3;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint16_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint32_t, src_stride, src_line, 1);

    while (height--) {
        dst = dst_line;
        src = src_line;

        dst_line += dst_stride;
        src_line += src_stride;
        w = width;

        while (w >= 16) {
            vsrc = __lasx_xvld(src, 0);
            vdst = __lasx_xvld(dst, 0);
            w -= 16;

            unpack_565_256_4x256(vdst, &vdst0, &vdst1, &vdst2, &vdst3);
            dst0 = __lasx_xvpermi_q(vdst2, vdst0, 0x20);
            dst1 = __lasx_xvpermi_q(vdst3, vdst1, 0x20);
            dst2 = __lasx_xvpermi_q(vdst2, vdst0, 0x31);
            dst3 = __lasx_xvpermi_q(vdst3, vdst1, 0x31);

            unpack_256_2x256((__m256i)vsrc, (__m256i*)&vsrc_lo, (__m256i*)&vsrc_hi);
            expand_alpha_2x256(vsrc_lo, vsrc_hi, &valpha_lo, &valpha_hi);
            over_2x256(&vsrc_lo, &vsrc_hi, &valpha_lo, &valpha_hi, &dst0, &dst1);

            vsrc = __lasx_xvld(src, 32);
            unpack_256_2x256((__m256i)vsrc, (__m256i*)&vsrc_lo, (__m256i*)&vsrc_hi);
            expand_alpha_2x256(vsrc_lo, vsrc_hi, &valpha_lo, &valpha_hi);
            over_2x256(&vsrc_lo, &vsrc_hi, &valpha_lo, &valpha_hi, &dst2, &dst3);

            vdst0 = __lasx_xvpermi_q(dst2, dst0, 0x20);
            vdst1 = __lasx_xvpermi_q(dst3, dst1, 0x20);
            vdst2 = __lasx_xvpermi_q(dst2, dst0, 0x31);
            vdst3 = __lasx_xvpermi_q(dst3, dst1, 0x31);

            __lasx_xvst(pack_565_4x256_256(&vdst0, &vdst1, &vdst2, &vdst3), dst, 0);

            dst += 16;
            src += 16;
        }

        while (w--) {
            s = *src++;
            d = *dst;
            *dst++ = composite_over_8888_0565pixel(s, d);
        }
    }
}

static void
lasx_composite_over_n_8_0565 (pixman_implementation_t *imp,
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

    __m256i mask_zero = __lasx_xvldi(0);
    __m256i lasx_src, lasx_alpha, lasx_mask, lasx_dest;
    __m256i vsrc, valpha;
    __m256i vmask, vmaska, vmask_lo, vmask_hi;
    __m256i vdst, vdst0, vdst1, vdst2, vdst3;
    __m256i dst0, dst1, dst2, dst3;

    src = _pixman_image_get_solid (imp, src_image, dest_image->bits.format);

    if (src == 0)
        return;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint16_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (mask_image, mask_x, mask_y, uint8_t, mask_stride, mask_line, 1);

    lasx_src = expand_pixel_32_1x256(src);
    lasx_alpha = expand_alpha_1x256(lasx_src);

    vsrc = expand_pixel_32_2x256(src);
    valpha = expand_alphaa_2x256(vsrc);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        mask = (void*)mask_line;
        mask_line += mask_stride;
        w = width;

        while (w >= 16) {
            vdst = __lasx_xvld(dst, 0);
            w -= 16;
            unpack_565_256_4x256(vdst, &vdst0, &vdst1, &vdst2, &vdst3);
            dst0 = __lasx_xvpermi_q(vdst2, vdst0, 0x20);
            dst1 = __lasx_xvpermi_q(vdst3, vdst1, 0x20);
            dst2 = __lasx_xvpermi_q(vdst2, vdst0, 0x31);
            dst3 = __lasx_xvpermi_q(vdst3, vdst1, 0x31);

            m = *mask;
            vmaska = unpack_32_1x256(m);
            mask += 1;
            m = *mask;
            vmask = unpack_32_1x256(m);
            vmask = __lasx_xvpermi_q(vmask, vmaska, 0x20);
            mask += 1;
            vmask = __lasx_xvilvl_b(mask_zero, vmask);

            unpack_256_2x256(vmask, (__m256i*)&vmask_lo, (__m256i*)&vmask_hi);
            expand_alpha_rev_2x256(vmask_lo, vmask_hi, &vmask_lo, &vmask_hi);
            in_over_2x256(&vsrc, &vsrc, &valpha, &valpha, &vmask_lo, &vmask_hi,
                          &dst0, &dst1);

            m = *mask;
            vmaska = unpack_32_1x256(m);
            mask += 1;
            m = *mask;
            vmask = unpack_32_1x256(m);
            vmask = __lasx_xvpermi_q(vmask, vmaska, 0x20);
            mask += 1;
            vmask = __lasx_xvilvl_b(mask_zero, vmask);

            unpack_256_2x256(vmask, (__m256i*)&vmask_lo, (__m256i*)&vmask_hi);
            expand_alpha_rev_2x256(vmask_lo, vmask_hi, &vmask_lo, &vmask_hi);
            in_over_2x256(&vsrc, &vsrc, &valpha, &valpha, &vmask_lo, &vmask_hi,
                          &dst2, &dst3);

            vdst0 = __lasx_xvpermi_q(dst2, dst0, 0x20);
            vdst1 = __lasx_xvpermi_q(dst3, dst1, 0x20);
            vdst2 = __lasx_xvpermi_q(dst2, dst0, 0x31);
            vdst3 = __lasx_xvpermi_q(dst3, dst1, 0x31);

            __lasx_xvst(pack_565_4x256_256(&vdst0, &vdst1, &vdst2, &vdst3), dst, 0);

            dst += 16;
        }

        p = (void*)mask;
        while (w--) {
            m = *p++;

            if (m) {
                d = *dst;
                lasx_mask = expand_alpha_rev_1x256(unpack_32_1x256 (m));
                lasx_dest = expand565_16_1x256(d);

                *dst = pack_565_32_16(pack_1x256_32(in_over_1x256 (&lasx_src,
                                      &lasx_alpha, &lasx_mask, &lasx_dest)));
            }
            dst++;
        }
    }
}

static void
lasx_composite_over_x888_8_8888 (pixman_implementation_t *imp,
                                 pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t *src, *src_line, s;
    uint32_t *dst, *dst_line, d;
    uint8_t  *mask_line, *p;
    uint32_t *mask;
    uint32_t m, w;
    int src_stride, mask_stride, dst_stride;

    __m256i mask_zero = __lasx_xvldi(0);
    __m256i mask_4x32 = mask_ff000000;
    __m256i vsrc, vsrc_lo, vsrc_hi;
    __m256i vdst, vdst_lo, vdst_hi;
    __m256i vmask, vmask_lo, vmask_hi, vmaska;

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

        while (w >= 8) {
            m = *mask;
            vsrc = __lasx_xvld(src, 0);
            src += 8;
            w -= 8;
            vsrc |= mask_4x32;

            if (m == 0xffffffff) {
                __lasx_xvst(vsrc, dst, 0);
            } else {
                vdst = __lasx_xvld(dst, 0);
                vmask = __lasx_xvilvl_b(mask_zero, unpack_32_1x256(m));
                m = *(mask + 1);
                vmaska = __lasx_xvilvl_b(mask_zero, unpack_32_1x256(m));
                vmask = __lasx_xvpermi_q(vmaska, vmask, 0x20);

                unpack_256_2x256(vsrc, (__m256i*)&vsrc_lo, (__m256i*)&vsrc_hi);
                unpack_256_2x256(vmask, (__m256i*)&vmask_lo, (__m256i*)&vmask_hi);
                expand_alpha_rev_2x256(vmask_lo, vmask_hi, &vmask_lo, &vmask_hi);
                unpack_256_2x256(vdst, (__m256i*)&vdst_lo, (__m256i*)&vdst_hi);

                in_over_2x256(&vsrc_lo, &vsrc_hi, &mask_00ff, &mask_00ff,
                              &vmask_lo, &vmask_hi, &vdst_lo, &vdst_hi);

                __lasx_xvst(pack_2x256_256(vdst_lo, vdst_hi), dst, 0);
            }
            dst += 8;
            mask += 2;
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
                    __m256i ma, md, ms;
                    d = *dst;
                    ma = expand_alpha_rev_1x256(unpack_32_1x256(m));
                    md = unpack_32_1x256(d);
                    ms = unpack_32_1x256(s);
                    *dst = pack_1x256_32(in_over_1x256(&ms, &mask_00ff, &ma, &md));
                }
            }
            src++;
            dst++;
        }
    }
}

static void
lasx_composite_over_8888_n_8888 (pixman_implementation_t *imp,
                                 pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t *dst_line, *dst;
    uint32_t *src_line, *src;
    uint32_t mask, maska;
    int32_t w;
    int dst_stride, src_stride;

    __m256i vmask;
    __m256i vsrc, vsrc_lo, vsrc_hi;
    __m256i vdst, vdst_lo, vdst_hi;
    __m256i valpha_lo, valpha_hi;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint32_t, src_stride, src_line, 1);

    mask = _pixman_image_get_solid (imp, mask_image, PIXMAN_a8r8g8b8);
    maska = mask >> 24;
    vmask = __lasx_xvreplgr2vr_h(maska);

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        src = src_line;
        src_line += src_stride;
        w = width;

        while (w && (uintptr_t)dst & 15) {
            uint32_t s = *src++;

            if (s) {
                uint32_t d = *dst;
                __m256i ms = unpack_32_1x256(s);
                __m256i alpha = expand_alpha_1x256(ms);
                __m256i mask = vmask;
                __m256i dest = unpack_32_1x256(d);
                *dst = pack_1x256_32(in_over_1x256(&ms, &alpha, &mask, &dest));
            }
            dst++;
            w--;
        }

        while (w >= 8) {
            vsrc = __lasx_xvld(src, 0);
            src += 8;
            w -= 8;

            if (__lasx_xbnz_v(vsrc)) {
                vdst = __lasx_xvld(dst, 0);
                unpack_256_2x256(vsrc, (__m256i*)&vsrc_lo, (__m256i*)&vsrc_hi);
                unpack_256_2x256(vdst, (__m256i*)&vdst_lo, (__m256i*)&vdst_hi);
                expand_alpha_2x256(vsrc_lo, vsrc_hi,  &valpha_lo, &valpha_hi);

                in_over_2x256(&vsrc_lo, &vsrc_hi, &valpha_lo, &valpha_hi,
                              &vmask, &vmask, &vdst_lo, &vdst_hi);
                __lasx_xvst(pack_2x256_256(vdst_lo, vdst_hi), dst, 0);
            }
            dst += 8;
        }

        if (w >= 4) {
            vsrc_lo = __lasx_xvldrepl_d(src, 0);
            vsrc_hi = __lasx_xvldrepl_d(src, 8);
            vsrc  = __lasx_xvilvl_d(vsrc_hi, vsrc_lo);
            src += 4;
            w -= 4;

            if (__lasx_xbnz_v(vsrc)) {
                vdst_lo = __lasx_xvldrepl_d(dst, 0);
                vdst_hi = __lasx_xvldrepl_d(dst, 8);
                vdst  = __lasx_xvilvl_d(vdst_hi, vdst_lo);
                unpack_256_2x256(vsrc, (__m256i*)&vsrc_lo, (__m256i*)&vsrc_hi);
                unpack_256_2x256(vdst, (__m256i*)&vdst_lo, (__m256i*)&vdst_hi);
                expand_alpha_2x256(vsrc_lo, vsrc_hi,  &valpha_lo, &valpha_hi);
                in_over_2x256(&vsrc_lo, &vsrc_hi, &valpha_lo, &valpha_hi,
                              &vmask, &vmask, &vdst_lo, &vdst_hi);
                vdst = pack_2x256_256(vdst_lo, vdst_hi);
                __lasx_xvstelm_d(vdst, dst, 0, 0);
                __lasx_xvstelm_d(vdst, dst, 8, 1);
            }
            dst += 4;
        }

        while (w--) {
            uint32_t s = *src++;

            if (s) {
                uint32_t d = *dst;
                __m256i ms = unpack_32_1x256(s);
                __m256i alpha = expand_alpha_1x256(ms);
                __m256i mask = vmask;
                __m256i dest = unpack_32_1x256(d);
                *dst = pack_1x256_32(in_over_1x256(&ms, &alpha, &mask, &dest));
            }
            dst++;
        }
    }
}

static void
lasx_composite_over_x888_n_8888 (pixman_implementation_t *imp,
                                 pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t *dst_line, *dst;
    uint32_t *src_line, *src;
    uint32_t mask, maska;
    int dst_stride, src_stride;
    int32_t w;

    __m256i vmask, valpha, mask_4x32;
    __m256i vsrc, vsrc_lo, vsrc_hi;
    __m256i vdst, vdst_lo, vdst_hi;
    __m256i zero = __lasx_xvldi(0);

    mask_4x32 = __lasx_xvreplgr2vr_w(0xff000000);

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint32_t, src_stride, src_line, 1);

    mask = _pixman_image_get_solid (imp, mask_image, PIXMAN_a8r8g8b8);
    maska = mask >> 24;

    vmask = __lasx_xvreplgr2vr_h(maska);
    valpha = mask_00ff;

    while (height--) {
        dst = dst_line;
        dst_line += dst_stride;
        src = src_line;
        src_line += src_stride;
        w = width;

        while (w >= 8) {
            vsrc = __lasx_xvld(src, 0);
            src += 8;
            w -= 8;
            vsrc = __lasx_xvor_v(vsrc, mask_4x32);
            vdst = __lasx_xvld(dst, 0);

            unpack_256_2x256(vsrc, (__m256i*)&vsrc_lo, (__m256i*)&vsrc_hi);
            unpack_256_2x256(vdst, (__m256i*)&vdst_lo, (__m256i*)&vdst_hi);

            in_over_2x256(&vsrc_lo, &vsrc_hi, &valpha, &valpha,
                          &vmask, &vmask, &vdst_lo, &vdst_hi);

            __lasx_xvst(pack_2x256_256(vdst_lo, vdst_hi), dst, 0);
            dst += 8;
        }

        while (w--) {
            uint32_t s = (*src++) | 0xff000000;
            uint32_t d = *dst;

            __m256i alpha, tmask;
            __m256i src = unpack_32_1x256 (s);
            __m256i dest  = unpack_32_1x256 (d);

            alpha = __lasx_xvpermi_q(zero, valpha, 0x20);
            tmask = __lasx_xvpermi_q(zero, vmask, 0x20);

            *dst = pack_1x256_32(in_over_1x256(&src,  &alpha, &tmask, &dest));

            dst++;
        }
    }
}

static void
lasx_composite_over_n_8888_0565_ca (pixman_implementation_t *imp,
                                    pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t src;
    uint16_t *dst_line, *dst, d;
    uint32_t *mask_line, *mask, m;
    int dst_stride, mask_stride;
    int w, flag;

    __m256i vsrc, valpha;
    __m256i lasx_src, lasx_alpha, lasx_mask, lasx_dest;
    __m256i vmask, vmask_lo, vmask_hi;
    __m256i vdst, vdst0, vdst1, vdst2, vdst3;
    __m256i dst0, dst1, dst2, dst3;

    src = _pixman_image_get_solid (imp, src_image, dest_image->bits.format);

    if (src == 0)
        return;

    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint16_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (mask_image, mask_x, mask_y, uint32_t, mask_stride, mask_line, 1);

    lasx_src = expand_pixel_32_1x256(src);
    lasx_alpha = expand_alpha_1x256(lasx_src);

    vsrc = expand_pixel_32_2x256(src);
    valpha = expand_alphaa_2x256(vsrc);

    while (height--) {
        mask = mask_line;
        dst = dst_line;
        mask_line += mask_stride;
        dst_line += dst_stride;
        w = width;

        while (w >= 16) {
            vmask = __lasx_xvld(mask, 0);
            vdst = __lasx_xvld(dst, 0);
            w -= 16;

            unpack_565_256_4x256(vdst, &vdst0, &vdst1, &vdst2, &vdst3);
            dst0 = __lasx_xvpermi_q(vdst2, vdst0, 0x20);
            dst1 = __lasx_xvpermi_q(vdst3, vdst1, 0x20);
            dst2 = __lasx_xvpermi_q(vdst2, vdst0, 0x31);
            dst3 = __lasx_xvpermi_q(vdst3, vdst1, 0x31);

            flag = __lasx_xbnz_v(vmask);
            unpack_256_2x256(vmask, (__m256i*)&vmask_lo, (__m256i*)&vmask_hi);
            vmask = __lasx_xvld(mask, 32);
            if (flag) {
                in_over_2x256(&vsrc, &vsrc, &valpha, &valpha, &vmask_lo, &vmask_hi,
                              &dst0, &dst1);
            }

            flag = __lasx_xbnz_v(vmask);
            unpack_256_2x256(vmask, (__m256i*)&vmask_lo, (__m256i*)&vmask_hi);
            if (flag) {
                in_over_2x256(&vsrc, &vsrc, &valpha, &valpha, &vmask_lo, &vmask_hi,
                              &dst2, &dst3);
            }

            vdst0 = __lasx_xvpermi_q(dst2, dst0, 0x20);
            vdst1 = __lasx_xvpermi_q(dst3, dst1, 0x20);
            vdst2 = __lasx_xvpermi_q(dst2, dst0, 0x31);
            vdst3 = __lasx_xvpermi_q(dst3, dst1, 0x31);

            __lasx_xvst(pack_565_4x256_256(&vdst0, &vdst1, &vdst2, &vdst3), dst, 0);
            dst += 16;
            mask += 16;
        }

        while (w--) {
            m = *(uint32_t *) mask;

            if (m) {
                d = *dst;
                lasx_mask = unpack_32_1x256(m);
                lasx_dest = expand565_16_1x256(d);
                *dst = pack_565_32_16(pack_1x256_32(in_over_1x256(&lasx_src, &lasx_alpha,
                                      &lasx_mask, &lasx_dest)));
            }
            dst++;
            mask++;
        }
    }
}

static uint32_t *
lasx_fetch_x8r8g8b8 (pixman_iter_t *iter, const uint32_t *mask)
{
    __m256i mask_4x32 = mask_ff000000;
    int w = iter->width;
    uint32_t *dst = iter->buffer;
    uint32_t *src = (uint32_t *)iter->bits;
    iter->bits += iter->stride;

    while (w >= 8) {
        __lasx_xvst(__lasx_xvor_v(__lasx_xvld(src, 0), mask_4x32), dst, 0);
        dst += 8;
        src += 8;
        w   -= 8;
    }

    while (w--) {
        *dst++ = (*src++) | 0xff000000;
    }

    return iter->buffer;
}

static uint32_t *
lasx_fetch_r5g6b5 (pixman_iter_t *iter, const uint32_t *mask)
{
    __m256i a, sa, s0, s1, s2, s3, s4;
    __m256i mask_red, mask_green, mask_blue;
    __m256i tmp0, tmp1;

    int w = iter->width;
    uint32_t *dst = iter->buffer;
    uint16_t *src = (uint16_t *)iter->bits;
    iter->bits += iter->stride;

    mask_red = __lasx_xvreplgr2vr_h(248);
    mask_green = __lasx_xvreplgr2vr_h(252);
    mask_blue = mask_red;
    a = __lasx_xvreplgr2vr_h(255) << 8;

    while (w >= 16) {
        s0 = __lasx_xvld(src, 0);
        src += 16;
        w   -= 16;

        //r
        s1 = __lasx_xvsrli_h(s0, 8);
        s1 &= mask_red;
        s2 = __lasx_xvsrli_h(s1, 5);
        s1 |= s2;

        //g
        s2 = __lasx_xvsrli_h(s0, 3);
        s2 &= mask_green;
        s3 = __lasx_xvsrli_h(s2, 6);
        s2 |= s3;

        //b
        s3 = s0 << 3;
        s3 &= mask_blue;
        s4 = __lasx_xvsrli_h(s3, 5);
        s3 |= s4;

        //ar
        sa = a | s1;

        //gb
        s2 <<= 8;
        s2 |= s3;

        tmp0 = __lasx_xvilvl_h(sa, s2);
        tmp1 = __lasx_xvilvh_h(sa, s2);
        s1   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        s3   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        __lasx_xvst(s1, dst, 0);
        __lasx_xvst(s3, dst, 32);
        dst += 16;
    }

    while (w--) {
        uint16_t s = *src++;
        *dst++ = convert_0565_to_8888(s);
    }

    return iter->buffer;
}

static uint32_t *
lasx_fetch_a8 (pixman_iter_t *iter, const uint32_t *mask)
{
    __m256i srcv;
    __m256i t0, t1, t2, t3;
    __m256i dst0, dst1, dst2, dst3;
    __m256i zero = __lasx_xvldi(0);
    int w = iter->width;
    uint32_t *dst = iter->buffer;
    uint8_t *src = iter->bits;

    iter->bits += iter->stride;

    while (w >= 32) {
        srcv = __lasx_xvld(src, 0);
        src += 32;
        w   -= 32;
        dst0 = __lasx_xvilvl_b(srcv, zero);
        dst1 = __lasx_xvilvh_b(srcv, zero);
        t0 = __lasx_xvilvl_h(dst0, zero);
        t1 = __lasx_xvilvh_h(dst0, zero);
        t2 = __lasx_xvilvl_h(dst1, zero);
        t3 = __lasx_xvilvh_h(dst1, zero);
        dst0 = __lasx_xvpermi_q(t1, t0, 0x20);
        dst1 = __lasx_xvpermi_q(t3, t2, 0x20);
        dst2 = __lasx_xvpermi_q(t1, t0, 0x31);
        dst3 = __lasx_xvpermi_q(t3, t2, 0x31);
        __lasx_xvst(dst0, dst, 0);
        __lasx_xvst(dst1, dst, 32);
        __lasx_xvst(dst2, dst, 64);
        __lasx_xvst(dst3, dst, 96);
        dst += 32;
    }

    while (w--) {
        *dst++ = *(src++) << 24;
    }

    return iter->buffer;
}

// fetch/store 8 bits
static void lasx_fetch_scanline_a8 (bits_image_t *image, int x, int y, int width,
                                    uint32_t *buffer, const uint32_t *mask)
{
    uint8_t *bits = (uint8_t *)(image->bits + y * image->rowstride);
    __m256i src;
    __m256i t0, t1;
    __m256i temp0, temp1, temp2, temp3;
    __m256i dst0, dst1, dst2, dst3;
    __m256i zero = __lasx_xvldi(0);
    bits += x;

    while (width >= 32) {
        src = __lasx_xvld(bits, 0);
        t0 = __lasx_xvilvl_b(src, zero);
        t1 = __lasx_xvilvh_b(src, zero);
        temp0 = __lasx_xvilvl_h(t0, zero);
        temp1 = __lasx_xvilvh_h(t0, zero);
        temp2 = __lasx_xvilvl_h(t1, zero);
        temp3 = __lasx_xvilvh_h(t1, zero);
        dst0 = __lasx_xvpermi_q(temp1, temp0, 0x20);
        dst1 = __lasx_xvpermi_q(temp3, temp2, 0x20);
        dst2 = __lasx_xvpermi_q(temp1, temp0, 0x31);
        dst3 = __lasx_xvpermi_q(temp3, temp2, 0x31);
        __lasx_xvst(dst0, buffer, 0);
        __lasx_xvst(dst1, buffer, 32);
        __lasx_xvst(dst2, buffer, 64);
        __lasx_xvst(dst3, buffer, 96);
        bits += 32, width -= 32, buffer += 32;
    }

    if (width >= 8) {
        src = __lasx_xvldrepl_d(bits, 0);
        t0 = __lasx_xvilvl_b(src, zero);
        t0  = __lasx_xvpermi_d(t0, 0xd8);
        dst0 = __lasx_xvilvl_h(t0, zero);
        __lasx_xvst(dst0, buffer, 0);
        bits += 8; width -= 8; buffer += 8;
    }

    while (width--) {
        *buffer++ = ((*bits++) << 24);
    }
}

static void lasx_store_scanline_a8 (bits_image_t *image, int x, int y, int width,
                                    const uint32_t *values)
{
    uint8_t *dest = (uint8_t *)(image->bits + y * image->rowstride);
    __m256i src0, src1, src2, src3;
    __m256i cont  = {0x0000000400000000, 0x0000000500000001, 0x0000000600000002, 0x0000000700000003};
    dest += x;
    while (width >= 32) {
        src0 = __lasx_xvld(values, 0);
        src1 = __lasx_xvld(values, 32);
        src2 = __lasx_xvld(values, 64);
        src3 = __lasx_xvld(values, 96);
        src0 = __lasx_xvsrli_w(src0, 24);
        src1 = __lasx_xvsrli_w(src1, 24);
        src2 = __lasx_xvsrli_w(src2, 24);
        src3 = __lasx_xvsrli_w(src3, 24);
        src0 = __lasx_xvpickev_h(src1, src0);
        src1 = __lasx_xvpickev_h(src3, src2);
        src0 = __lasx_xvpickev_b(src1, src0);
        src0 = __lasx_xvperm_w(src0, cont);
        __lasx_xvst(src0, dest, 0);
        values += 32, width -= 32, dest += 32;
    }

    if (width >= 16) {
        src0 = __lasx_xvld(values, 0);
        src1 = __lasx_xvld(values, 32);
        src0 = __lasx_xvsrli_w(src0, 24);
        src1 = __lasx_xvsrli_w(src1, 24);
        src0 = __lasx_xvpickev_h(src1, src0);
        src0 = __lasx_xvpickev_b(src0, src0);
        src0 = __lasx_xvperm_w(src0, cont);
        __lasx_xvstelm_d(src0, dest, 0, 0);
        __lasx_xvstelm_d(src0, dest, 8, 1);
        values += 16; width -= 16; dest += 16;
    }

    if (width >= 8) {
        src0 = __lasx_xvld(values, 0);
        src0 = __lasx_xvsrli_w(src0, 24);
        src1 = __lasx_xvpermi_q(src0, src0, 0x01);
        src0 = __lasx_xvpickev_h(src1, src0);
        src0 = __lasx_xvpickev_b(src0, src0);
        __lasx_xvstelm_d(src0, dest, 0, 0);
        values += 8; width -= 8; dest += 8;
    }

    while (width--) {
        *dest++ = ((*values++) >> 24);
    }
}

static void lasx_fetch_scanline_a2r2g2b2 (bits_image_t *image, int x, int y,
                                          int width, uint32_t *buffer,
                                          const uint32_t *mask)
{
    uint8_t *bits = (uint8_t *)(image->bits + y * image->rowstride);
    uint32_t pixel, pixel0, pixel1, pixel2, pixel3;

    __m256i src;
    __m256i t0, t1, t2, t3, t4, t5, t6, t7;
    __m256i temp0, temp1, temp2, temp3;
    __m256i mask0 = __lasx_xvreplgr2vr_b(0xc0);
    __m256i mask1 = __lasx_xvreplgr2vr_b(0x30);
    __m256i mask2 = __lasx_xvreplgr2vr_b(0x0c);
    __m256i mask3 = __lasx_xvreplgr2vr_b(0x03);
    bits += x;

    while (width >= 32) {
        src = __lasx_xvld(bits, 0);
        t0 = (src & mask0); t1 = (src & mask1);
        t2 = (src & mask2); t3 = (src & mask3);
        t0 |= __lasx_xvsrli_b(t0, 2), t0 |= __lasx_xvsrli_b(t0, 4);
        t1 |= __lasx_xvslli_b(t1, 2), t1 |= __lasx_xvsrli_b(t1, 4);
        t2 |= __lasx_xvsrli_b(t2, 2), t2 |= __lasx_xvslli_b(t2, 4);
        t3 |= __lasx_xvslli_b(t3, 2), t3 |= __lasx_xvslli_b(t3, 4);
        t4 = __lasx_xvilvl_b(t0, t1);
        t5 = __lasx_xvilvh_b(t0, t1);
        t6 = __lasx_xvilvl_b(t2, t3);
        t7 = __lasx_xvilvh_b(t2, t3);
        t0 = __lasx_xvilvl_h(t4, t6);
        t1 = __lasx_xvilvh_h(t4, t6);
        t2 = __lasx_xvilvl_h(t5, t7);
        t3 = __lasx_xvilvh_h(t5, t7);
        temp0 = __lasx_xvpermi_q(t1, t0, 0x20);
        temp1 = __lasx_xvpermi_q(t3, t2, 0x20);
        temp2 = __lasx_xvpermi_q(t1, t0, 0x31);
        temp3 = __lasx_xvpermi_q(t3, t2, 0x31);
        __lasx_xvst(temp0, buffer, 0);
        __lasx_xvst(temp1, buffer, 32);
        __lasx_xvst(temp2, buffer, 64);
        __lasx_xvst(temp3, buffer, 96);
        bits += 32, width -= 32, buffer += 32;
    }

    if (width >= 8) {
        src = __lasx_xvldrepl_d(bits, 0);
        t0 = (src & mask0); t1 = (src & mask1);
        t2 = (src & mask2); t3 = (src & mask3);
        t0 |= __lasx_xvsrli_b(t0, 2), t0 |= __lasx_xvsrli_b(t0, 4);
        t1 |= __lasx_xvslli_b(t1, 2), t1 |= __lasx_xvsrli_b(t1, 4);
        t2 |= __lasx_xvsrli_b(t2, 2), t2 |= __lasx_xvslli_b(t2, 4);
        t3 |= __lasx_xvslli_b(t3, 2), t3 |= __lasx_xvslli_b(t3, 4);
        t4 = __lasx_xvilvl_b(t0, t1);
        t5 = __lasx_xvilvl_b(t2, t3);
        t4 = __lasx_xvpermi_d(t4, 0xd8);
        t5 = __lasx_xvpermi_d(t5, 0xd8);
        t0 = __lasx_xvilvl_h(t4, t5);
        __lasx_xvst(t0, buffer, 0);
        bits += 8; width -= 8; buffer += 8;
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

static void lasx_store_scanline_a2r2g2b2 (bits_image_t *image, int x, int y,
                                          int width, const uint32_t *values)
{
    uint8_t *dest = (uint8_t *)(image->bits + y * image->rowstride);
    __m256i in0, in1, in2, in3, in4, in5, in6, in7;
    __m256i in8, in9, in10, in11, in12, in13, in14, in15;
    __m256i tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;
    __m256i tt8, tt9, tt10, tt11, tt12, tt13, tt14, tt15;
    __m256i d0, d1;
    __m256i mask = __lasx_xvreplgr2vr_b(0xc0);

    dest += x;

    while (width >= 128) {
        in0 = __lasx_xvld(values, 0);
        in1 = __lasx_xvld(values, 32);
        in2 = __lasx_xvld(values, 64);
        in3 = __lasx_xvld(values, 96);
        in4 = __lasx_xvld(values, 128);
        in5 = __lasx_xvld(values, 160);
        in6 = __lasx_xvld(values, 192);
        in7 = __lasx_xvld(values, 224);
        values += 64;
        in8  = __lasx_xvld(values, 0);
        in9  = __lasx_xvld(values, 32);
        in10 = __lasx_xvld(values, 64);
        in11 = __lasx_xvld(values, 96);
        in12 = __lasx_xvld(values, 128);
        in13 = __lasx_xvld(values, 160);
        in14 = __lasx_xvld(values, 192);
        in15 = __lasx_xvld(values, 224);

        tt0  = __lasx_xvpermi_q(in8,  in0, 0x20);
        tt2  = __lasx_xvpermi_q(in9,  in1, 0x20);
        tt4  = __lasx_xvpermi_q(in10, in2, 0x20);
        tt6  = __lasx_xvpermi_q(in11, in3, 0x20);
        tt8  = __lasx_xvpermi_q(in12, in4, 0x20);
        tt10 = __lasx_xvpermi_q(in13, in5, 0x20);
        tt12 = __lasx_xvpermi_q(in14, in6, 0x20);
        tt14 = __lasx_xvpermi_q(in15, in7, 0x20);

        tt1  = __lasx_xvpermi_q(in8,  in0, 0x31);
        tt3  = __lasx_xvpermi_q(in9,  in1, 0x31);
        tt5  = __lasx_xvpermi_q(in10, in2, 0x31);
        tt7  = __lasx_xvpermi_q(in11, in3, 0x31);
        tt9  = __lasx_xvpermi_q(in12, in4, 0x31);
        tt11 = __lasx_xvpermi_q(in13, in5, 0x31);
        tt13 = __lasx_xvpermi_q(in14, in6, 0x31);
        tt15 = __lasx_xvpermi_q(in15, in7, 0x31);

        LASX_TRANSPOSE16x8_H(tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7,
                             tt8, tt9, tt10, tt11, tt12, tt13, tt14, tt15,
                             in0, in1, in2, in3, in4, in5, in6, in7);
        in8  = __lasx_xvpickev_b(in4, in0);
        in8  = __lasx_xvpermi_d(in8, 0xd8);
        in9  = __lasx_xvpickod_b(in4, in0);
        in9  = __lasx_xvpermi_d(in9, 0xd8);
        in10 = __lasx_xvpickev_b(in5, in1);
        in10 = __lasx_xvpermi_d(in10, 0xd8);
        in11 = __lasx_xvpickod_b(in5, in1);
        in11 = __lasx_xvpermi_d(in11, 0xd8);
        in12 = __lasx_xvpickev_b(in6, in2);
        in12 = __lasx_xvpermi_d(in12, 0xd8);
        in13 = __lasx_xvpickod_b(in6, in2);
        in13 = __lasx_xvpermi_d(in13, 0xd8);
        in14 = __lasx_xvpickev_b(in7, in3);
        in14 = __lasx_xvpermi_d(in14, 0xd8);
        in15 = __lasx_xvpickod_b(in7, in3);
        in15 = __lasx_xvpermi_d(in15, 0xd8);

        in8 &= mask, in9 &= mask, in10 &= mask, in11 &= mask;
        in12 &= mask, in13 &= mask, in14 &= mask, in15 &= mask;
        in8 = __lasx_xvsrli_b(in8, 6), in12 = __lasx_xvsrli_b(in12, 6);
        in9 = __lasx_xvsrli_b(in9, 4), in13 = __lasx_xvsrli_b(in13, 4);
        in10 = __lasx_xvsrli_b(in10, 2), in14 = __lasx_xvsrli_b(in14, 2);
        d0 = in8, d0 |= in9, d0 |= in10, d0 |= in11;
        d1 = in12, d1 |= in13, d1 |= in14, d1 |= in15;

        tt0  = __lasx_xvpermi_q(tt0,  tt0,  0x31);
        tt1  = __lasx_xvpermi_q(tt1,  tt1,  0x31);
        tt2  = __lasx_xvpermi_q(tt2,  tt2,  0x31);
        tt3  = __lasx_xvpermi_q(tt3,  tt3,  0x31);
        tt4  = __lasx_xvpermi_q(tt4,  tt4,  0x31);
        tt5  = __lasx_xvpermi_q(tt5,  tt5,  0x31);
        tt6  = __lasx_xvpermi_q(tt6,  tt6,  0x31);
        tt7  = __lasx_xvpermi_q(tt7,  tt7,  0x31);
        tt8  = __lasx_xvpermi_q(tt8,  tt8,  0x31);
        tt9  = __lasx_xvpermi_q(tt9,  tt9,  0x31);
        tt10 = __lasx_xvpermi_q(tt10, tt10, 0x31);
        tt11 = __lasx_xvpermi_q(tt11, tt11, 0x31);
        tt12 = __lasx_xvpermi_q(tt12, tt12, 0x31);
        tt13 = __lasx_xvpermi_q(tt13, tt13, 0x31);
        tt14 = __lasx_xvpermi_q(tt14, tt14, 0x31);
        tt15 = __lasx_xvpermi_q(tt15, tt15, 0x31);

        LASX_TRANSPOSE16x8_H(tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7,
                             tt8, tt9, tt10, tt11, tt12, tt13, tt14, tt15,
                             in0, in1, in2, in3, in4, in5, in6, in7);
        in8  = __lasx_xvpickev_b(in4, in0);
        in8  = __lasx_xvpermi_d(in8, 0xd8);
        in9  = __lasx_xvpickod_b(in4, in0);
        in9  = __lasx_xvpermi_d(in9, 0xd8);
        in10 = __lasx_xvpickev_b(in5, in1);
        in10 = __lasx_xvpermi_d(in10, 0xd8);
        in11 = __lasx_xvpickod_b(in5, in1);
        in11 = __lasx_xvpermi_d(in11, 0xd8);
        in12 = __lasx_xvpickev_b(in6, in2);
        in12 = __lasx_xvpermi_d(in12, 0xd8);
        in13 = __lasx_xvpickod_b(in6, in2);
        in13 = __lasx_xvpermi_d(in13, 0xd8);
        in14 = __lasx_xvpickev_b(in7, in3);
        in14 = __lasx_xvpermi_d(in14, 0xd8);
        in15 = __lasx_xvpickod_b(in7, in3);
        in15 = __lasx_xvpermi_d(in15, 0xd8);

        in8 &= mask, in9 &= mask, in10 &= mask, in11 &= mask;
        in12 &= mask, in13 &= mask, in14 &= mask, in15 &= mask;
        in8 = __lasx_xvsrli_b(in8, 6), in12 = __lasx_xvsrli_b(in12, 6);
        in9 = __lasx_xvsrli_b(in9, 4), in13 = __lasx_xvsrli_b(in13, 4);
        in10 = __lasx_xvsrli_b(in10, 2), in14 = __lasx_xvsrli_b(in14, 2);
        tt0 = in8, tt0 |= in9, tt0 |= in10, tt0 |= in11;
        tt1 = in12, tt1 |= in13, tt1 |= in14, tt1 |= in15;

        in0 = __lasx_xvpermi_q(tt0, d0, 0x20);
        in2 = __lasx_xvpermi_q(tt0, d0, 0x31);
        in1 = __lasx_xvpermi_q(tt1, d1, 0x20);
        in3 = __lasx_xvpermi_q(tt1, d1, 0x31);

        in8  = __lasx_xvilvl_b(in1, in0);
        in9  = __lasx_xvilvh_b(in1, in0);
        in10 = __lasx_xvilvl_b(in3, in2);
        in11 = __lasx_xvilvh_b(in3, in2);

        in0 = __lasx_xvilvl_h(in10, in8);
        in1 = __lasx_xvilvh_h(in10, in8);
        in2 = __lasx_xvilvl_h(in11, in9);
        in3 = __lasx_xvilvh_h(in11, in9);

        d0  = __lasx_xvpermi_q(in1, in0, 0x20);
        tt0 = __lasx_xvpermi_q(in1, in0, 0x31);
        d1  = __lasx_xvpermi_q(in3, in2, 0x20);
        tt1 = __lasx_xvpermi_q(in3, in2, 0x31);

        __lasx_xvst(d0,  dest, 0);
        __lasx_xvst(d1,  dest, 32);
        __lasx_xvst(tt0, dest, 64);
        __lasx_xvst(tt1, dest, 96);
        width -= 128, values += 64, dest += 128;
    }

    while (width >= 32) {
        in0 = __lasx_xvld(values, 0);
        in2 = __lasx_xvld(values, 32);
        in4 = __lasx_xvld(values, 64);
        in6 = __lasx_xvld(values, 96);

        in1 = __lasx_xvpackod_d(in0, in0);
        in3 = __lasx_xvpackod_d(in2, in2);
        in5 = __lasx_xvpackod_d(in4, in4);
        in7 = __lasx_xvpackod_d(in6, in6);
        tt0 = __lasx_xvpermi_q(in4, in0, 0x20);
        tt2 = __lasx_xvpermi_q(in4, in0, 0x31);
        tt1 = __lasx_xvpermi_q(in5, in1, 0x20);
        tt3 = __lasx_xvpermi_q(in5, in1, 0x31);
        tt4 = __lasx_xvpermi_q(in6, in2, 0x20);
        tt6 = __lasx_xvpermi_q(in6, in2, 0x31);
        tt5 = __lasx_xvpermi_q(in7, in3, 0x20);
        tt7 = __lasx_xvpermi_q(in7, in3, 0x31);

        LASX_TRANSPOSE8x8_H(tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7,
                            tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7);
        tt8  = __lasx_xvpickev_b(tt1, tt0);
        tt8  = __lasx_xvpermi_d(tt8, 0xd8);
        tt9  = __lasx_xvpickod_b(tt1, tt0);
        tt9  = __lasx_xvpermi_d(tt9, 0xd8);
        tt10 = __lasx_xvpickev_b(tt3, tt2);
        tt10 = __lasx_xvpermi_d(tt10, 0xd8);
        tt11 = __lasx_xvpickod_b(tt3, tt2);
        tt11 = __lasx_xvpermi_d(tt11, 0xd8);
        tt12 = __lasx_xvpickev_b(tt5, tt4);
        tt12 = __lasx_xvpermi_d(tt12, 0xd8);
        tt13 = __lasx_xvpickod_b(tt5, tt4);
        tt13 = __lasx_xvpermi_d(tt13, 0xd8);
        tt14 = __lasx_xvpickev_b(tt7, tt6);
        tt14 = __lasx_xvpermi_d(tt14, 0xd8);
        tt15 = __lasx_xvpickod_b(tt7, tt6);
        tt15 = __lasx_xvpermi_d(tt15, 0xd8);

        tt0 = __lasx_xvpermi_q(tt12, tt8,  0x20);
        tt2 = __lasx_xvpermi_q(tt12, tt8,  0x31);
        tt1 = __lasx_xvpermi_q(tt13, tt9,  0x20);
        tt3 = __lasx_xvpermi_q(tt13, tt9,  0x31);
        tt4 = __lasx_xvpermi_q(tt14, tt10, 0x20);
        tt6 = __lasx_xvpermi_q(tt14, tt10, 0x31);
        tt5 = __lasx_xvpermi_q(tt15, tt11, 0x20);
        tt7 = __lasx_xvpermi_q(tt15, tt11, 0x31);

        tt0 &= mask, tt1 &= mask, tt2 &= mask, tt3 &= mask;
        tt4 &= mask, tt5 &= mask, tt6 &= mask, tt7 &= mask;
        tt0 = __lasx_xvsrli_b(tt0, 6), tt4 = __lasx_xvsrli_b(tt4, 6);
        tt1 = __lasx_xvsrli_b(tt1, 4), tt5 = __lasx_xvsrli_b(tt5, 4);
        tt2 = __lasx_xvsrli_b(tt2, 2), tt6 = __lasx_xvsrli_b(tt6, 2);
        d0 = tt0, d0 |= tt1, d0 |= tt2, d0 |= tt3;
        d1 = tt4, d1 |= tt5, d1 |= tt6, d1 |= tt7;

        tt0 = __lasx_xvilvl_b(d1, d0);
        tt1 = __lasx_xvilvh_b(d1, d0);
        d0  = __lasx_xvpermi_q(tt0, tt1, 0x02);
        __lasx_xvst(d0, dest, 0);
        width -= 32, values += 32, dest += 32;
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
static void lasx_fetch_scanline_a1r5g5b5 (bits_image_t *image, int x, int y,
                                          int width, uint32_t *buffer,
                                          const uint32_t *mask)
{
    uint16_t *bits = (uint16_t *)(image->bits + y * image->rowstride);
    uint32_t pixel, pixel0, pixel1, pixel2;

    __m256i src, tmp0, tmp1;
    __m256i t, t0, t1, t2, t3;
    __m256i mask0 = __lasx_xvreplgr2vr_h(0x001f);
    bits += x;

    while (width >= 16) {
        src  = __lasx_xvld(bits, 0);
        t0   = (src & mask0);
        t0   = __lasx_xvslli_h(t0, 3);
        t    = __lasx_xvsrli_h(t0, 5);
        t0  |= t;
        t1   = __lasx_xvsrli_h(src, 5);
        t1  &= mask0;
        t1   = __lasx_xvslli_h(t1, 3);
        t    = __lasx_xvsrli_h(t1, 5);
        t1  |= t;
        t2   = __lasx_xvsrli_h(src, 10);
        t2  &= mask0;
        t2   = __lasx_xvslli_h(t2, 3);
        t    = __lasx_xvsrli_h(t2, 5);
        t2  |= t;
        t3   = __lasx_xvsrli_h(src, 15);
        t    = __lasx_xvslli_h(t3, 1);
        t3  |= t;
        t    = __lasx_xvslli_h(t3, 2);
        t3  |= t;
        t    = __lasx_xvslli_h(t3, 4);
        t3  |= t;
        t1 <<= 8;
        t0  |= t1;
        t3 <<= 8;
        t2  |= t3;
        tmp0 = __lasx_xvilvl_h(t2, t0);
        tmp1 = __lasx_xvilvh_h(t2, t0);
        t1   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t3   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        __lasx_xvst(t1, buffer, 0);
        __lasx_xvst(t3, buffer, 32);
        bits += 16, width -= 16, buffer += 16;
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

static void lasx_store_scanline_a1r5g5b5 (bits_image_t *image, int x, int y,
                                         int width, const uint32_t *values)
{
    uint16_t *dest = (uint16_t *)(image->bits + y * image->rowstride);
    uint32_t pixel, pixel0, pixel1, pixel2, pixel3;
    __m256i in0, in1, in2, in3;
    __m256i tmp0, tmp1;
    __m256i t0, t1, t2, t3, t4, t5, t6, t7;
    __m256i t8, t9, t10, t11, t12, t13, t14, t15;
    __m256i zero = __lasx_xvldi(0);
    __m256i mask = { 0x80f8f8f880f8f8f8, 0x80f8f8f880f8f8f8,
                     0x80f8f8f880f8f8f8, 0x80f8f8f880f8f8f8 };
    dest += x;

    while (width >= 32) {
        in0 = __lasx_xvld(values, 0);
        in1 = __lasx_xvld(values, 32);
        in2 = __lasx_xvld(values, 64);
        in3 = __lasx_xvld(values, 96);

        in0  = __lasx_xvand_v(in0, mask);
        in1  = __lasx_xvand_v(in1, mask);
        in2  = __lasx_xvand_v(in2, mask);
        in3  = __lasx_xvand_v(in3, mask);
        tmp0 = __lasx_xvilvl_b(in0, zero);
        tmp1 = __lasx_xvilvh_b(in0, zero);
        t0   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t1   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_b(in1, zero);
        tmp1 = __lasx_xvilvh_b(in1, zero);
        t2   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t3   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_b(in2, zero);
        tmp1 = __lasx_xvilvh_b(in2, zero);
        t4   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t5   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_b(in3, zero);
        tmp1 = __lasx_xvilvh_b(in3, zero);
        t6   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t7   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);

        tmp0 = __lasx_xvilvl_h(zero, t7);
        tmp1 = __lasx_xvilvh_h(zero, t7);
        t14  = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t15  = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t6);
        tmp1 = __lasx_xvilvh_h(zero, t6);
        t12  = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t13  = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t5);
        tmp1 = __lasx_xvilvh_h(zero, t5);
        t10  = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t11  = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t4);
        tmp1 = __lasx_xvilvh_h(zero, t4);
        t8   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t9   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t3);
        tmp1 = __lasx_xvilvh_h(zero, t3);
        t6   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t7   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t2);
        tmp1 = __lasx_xvilvh_h(zero, t2);
        t4   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t5   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t1);
        tmp1 = __lasx_xvilvh_h(zero, t1);
        t2   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t3   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t0);
        tmp1 = __lasx_xvilvh_h(zero, t0);
        t0   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t1   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);

        LASX_TRANSPOSE8x8_W(t0, t1, t2, t3, t4, t5, t6, t7,
                            t0, t1, t2, t3, t4, t5, t6, t7);
        LASX_TRANSPOSE8x8_W(t8, t9, t10, t11, t12, t13, t14, t15,
                            t8, t9, t10, t11, t12, t13, t14, t15);

        t0 = __lasx_xvsrli_h(t0, 11);
        t1 = __lasx_xvsrli_h(t1, 6);
        t2 = __lasx_xvsrli_h(t2, 1);
        t4 = __lasx_xvsrli_h(t4, 11);
        t5 = __lasx_xvsrli_h(t5, 6);
        t6 = __lasx_xvsrli_h(t6, 1);

        t8  = __lasx_xvsrli_h(t8, 11);
        t9  = __lasx_xvsrli_h(t9, 6);
        t10 = __lasx_xvsrli_h(t10, 1);
        t12 = __lasx_xvsrli_h(t12, 11);
        t13 = __lasx_xvsrli_h(t13, 6);
        t14 = __lasx_xvsrli_h(t14, 1);

        t3 = __lasx_xvor_v(t3, t2);
        t3 = __lasx_xvor_v(t3, t1);
        t3 = __lasx_xvor_v(t3, t0);
        t7 = __lasx_xvor_v(t7, t6);
        t7 = __lasx_xvor_v(t7, t5);
        t7 = __lasx_xvor_v(t7, t4);

        t11 = __lasx_xvor_v(t11, t10);
        t11 = __lasx_xvor_v(t11, t9);
        t11 = __lasx_xvor_v(t11, t8);
        t15 = __lasx_xvor_v(t15, t14);
        t15 = __lasx_xvor_v(t15, t13);
        t15 = __lasx_xvor_v(t15, t12);

        tmp0 = __lasx_xvilvl_w(t7, t3);
        tmp1 = __lasx_xvilvh_w(t7, t3);
        t0   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t1   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        t0   = __lasx_xvpickev_h(t1, t0);
        t0   = __lasx_xvpermi_d(t0, 0xd8);

        tmp0 = __lasx_xvilvl_w(t15, t11);
        tmp1 = __lasx_xvilvh_w(t15, t11);
        t8   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t9   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        t1   = __lasx_xvpickev_h(t9, t8);
        t1   = __lasx_xvpermi_d(t1, 0xd8);

        __lasx_xvst(t0, dest, 0);
        __lasx_xvst(t1, dest, 32);
        values += 32, width -= 32, dest += 32;
    }

    if (width >= 16) {
        in0 = __lasx_xvld(values, 0);
        in1 = __lasx_xvld(values, 32);

        in0  = __lasx_xvand_v(in0, mask);
        in1  = __lasx_xvand_v(in1, mask);
        tmp0 = __lasx_xvilvl_b(in0, zero);
        tmp1 = __lasx_xvilvh_b(in0, zero);
        t0   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t1   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_b(in1, zero);
        tmp1 = __lasx_xvilvh_b(in1, zero);
        t2   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t3   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t3);
        tmp1 = __lasx_xvilvh_h(zero, t3);
        t6   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t7   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t2);
        tmp1 = __lasx_xvilvh_h(zero, t2);
        t4   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t5   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t1);
        tmp1 = __lasx_xvilvh_h(zero, t1);
        t2   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t3   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t0);
        tmp1 = __lasx_xvilvh_h(zero, t0);
        t0   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t1   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        LASX_TRANSPOSE8x8_W(t0, t1, t2, t3, t4, t5, t6, t7,
                            t0, t1, t2, t3, t4, t5, t6, t7);

        t0 = __lasx_xvsrli_h(t0, 11);
        t1 = __lasx_xvsrli_h(t1, 6);
        t2 = __lasx_xvsrli_h(t2, 1);
        t4 = __lasx_xvsrli_h(t4, 11);
        t5 = __lasx_xvsrli_h(t5, 6);
        t6 = __lasx_xvsrli_h(t6, 1);

        t3 = __lasx_xvor_v(t3, t2);
        t3 = __lasx_xvor_v(t3, t1);
        t3 = __lasx_xvor_v(t3, t0);
        t7 = __lasx_xvor_v(t7, t6);
        t7 = __lasx_xvor_v(t7, t5);
        t7 = __lasx_xvor_v(t7, t4);

        tmp0 = __lasx_xvilvl_w(t7, t3);
        tmp1 = __lasx_xvilvh_w(t7, t3);
        t0   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t1   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        t0   = __lasx_xvpickev_h(t1, t0);
        t0   = __lasx_xvpermi_d(t0, 0xd8);

        __lasx_xvst(t0, dest, 0);
        values += 16, width -= 16, dest += 16;
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

static void lasx_fetch_scanline_a4r4g4b4 (bits_image_t *image, int x, int y,
                                          int width, uint32_t *buffer,
                                          const uint32_t *mask)
{
    uint16_t *bits = (uint16_t *)(image->bits + y * image->rowstride);
    uint32_t pixel, pixel0, pixel1, pixel2;

    __m256i src, tmp0, tmp1;
    __m256i t, t0, t1, t2, t3;

    __m256i mask0 = __lasx_xvreplgr2vr_h(0x000f);
    bits += x;

    while (width >= 16) {
        src  = __lasx_xvld(bits, 0);
        t0   = __lasx_xvsrli_h(src, 12);
        t    = (t0 << 4), t0 |= t;
        t1   = __lasx_xvsrli_h(src, 8);
        t1  &= mask0, t = (t1 << 4), t1 |= t;
        t2   = __lasx_xvsrli_h(src, 4);
        t2  &= mask0, t = (t2 << 4), t2 |= t;
        t3   = (src & mask0), t = (t3 << 4), t3 |= t;
        t0 <<= 8, t2 <<= 8, t0 |= t1, t2 |= t3;
        tmp0 = __lasx_xvilvl_h(t0, t2);
        tmp1 = __lasx_xvilvh_h(t0, t2);
        t1   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t3   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        __lasx_xvst(t1, buffer, 0);
        __lasx_xvst(t3, buffer, 32);
        bits += 16, width -= 16, buffer += 16;
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

static void lasx_store_scanline_a4r4g4b4 (bits_image_t *image, int x, int y,
                                          int width, const uint32_t *values)
{
    uint16_t *dest = (uint16_t *)(image->bits + y * image->rowstride);
    uint32_t pixel, pixel0, pixel1;
    __m256i in0, in1, in2, in3;
    __m256i tmp0, tmp1;
    __m256i t0, t1, t2, t3, t4, t5, t6, t7;
    __m256i t8, t9, t10, t11, t12, t13, t14, t15;
    __m256i zero = __lasx_xvldi(0);
    __m256i mask = __lasx_xvreplgr2vr_h(0xf0f0);
    dest += x;

    while (width >= 32) {
        in0 = __lasx_xvld(values, 0);
        in1 = __lasx_xvld(values, 32);
        in2 = __lasx_xvld(values, 64);
        in3 = __lasx_xvld(values, 96);

        in0  = __lasx_xvand_v(in0, mask);
        in1  = __lasx_xvand_v(in1, mask);
        in2  = __lasx_xvand_v(in2, mask);
        in3  = __lasx_xvand_v(in3, mask);

        tmp0 = __lasx_xvilvl_b(in0, zero);
        tmp1 = __lasx_xvilvh_b(in0, zero);
        t0   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t1   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_b(in1, zero);
        tmp1 = __lasx_xvilvh_b(in1, zero);
        t2   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t3   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_b(in2, zero);
        tmp1 = __lasx_xvilvh_b(in2, zero);
        t4   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t5   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_b(in3, zero);
        tmp1 = __lasx_xvilvh_b(in3, zero);
        t6   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t7   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);

        tmp0 = __lasx_xvilvl_h(zero, t7);
        tmp1 = __lasx_xvilvh_h(zero, t7);
        t14  = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t15  = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t6);
        tmp1 = __lasx_xvilvh_h(zero, t6);
        t12  = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t13  = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t5);
        tmp1 = __lasx_xvilvh_h(zero, t5);
        t10  = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t11  = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t4);
        tmp1 = __lasx_xvilvh_h(zero, t4);
        t8   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t9   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t3);
        tmp1 = __lasx_xvilvh_h(zero, t3);
        t6   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t7   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t2);
        tmp1 = __lasx_xvilvh_h(zero, t2);
        t4   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t5   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t1);
        tmp1 = __lasx_xvilvh_h(zero, t1);
        t2   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t3   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t0);
        tmp1 = __lasx_xvilvh_h(zero, t0);
        t0   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t1   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);

        LASX_TRANSPOSE8x8_W(t0, t1, t2, t3, t4, t5, t6, t7,
                            t0, t1, t2, t3, t4, t5, t6, t7);
        LASX_TRANSPOSE8x8_W(t8, t9, t10, t11, t12, t13, t14, t15,
                            t8, t9, t10, t11, t12, t13, t14, t15);

        t0  = __lasx_xvsrli_h(t0, 12);
        t1  = __lasx_xvsrli_h(t1, 8);
        t2  = __lasx_xvsrli_h(t2, 4);
        t4  = __lasx_xvsrli_h(t4, 12);
        t5  = __lasx_xvsrli_h(t5, 8);
        t6  = __lasx_xvsrli_h(t6, 4);

        t8  = __lasx_xvsrli_h(t8, 12);
        t9  = __lasx_xvsrli_h(t9, 8);
        t10 = __lasx_xvsrli_h(t10, 4);
        t12 = __lasx_xvsrli_h(t12, 12);
        t13 = __lasx_xvsrli_h(t13, 8);
        t14 = __lasx_xvsrli_h(t14, 4);

        t3  = __lasx_xvor_v(t3, t2);
        t3  = __lasx_xvor_v(t3, t1);
        t3  = __lasx_xvor_v(t3, t0);
        t7  = __lasx_xvor_v(t7, t6);
        t7  = __lasx_xvor_v(t7, t5);
        t7  = __lasx_xvor_v(t7, t4);

        t11 = __lasx_xvor_v(t11, t10);
        t11 = __lasx_xvor_v(t11, t9);
        t11 = __lasx_xvor_v(t11, t8);
        t15 = __lasx_xvor_v(t15, t14);
        t15 = __lasx_xvor_v(t15, t13);
        t15 = __lasx_xvor_v(t15, t12);

        tmp0 = __lasx_xvilvl_w(t7, t3);
        tmp1 = __lasx_xvilvh_w(t7, t3);
        t0   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t1   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        t0   = __lasx_xvpickev_h(t1, t0);
        t0   = __lasx_xvpermi_d(t0, 0xd8);

        tmp0 = __lasx_xvilvl_w(t15, t11);
        tmp1 = __lasx_xvilvh_w(t15, t11);
        t8   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t9   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        t1   = __lasx_xvpickev_h(t9, t8);
        t1   = __lasx_xvpermi_d(t1, 0xd8);

        __lasx_xvst(t0, dest, 0);
        __lasx_xvst(t1, dest, 32);
        values += 32, width -= 32, dest += 32;
    }

    if (width >= 16) {

        in0 = __lasx_xvld(values, 0);
        in1 = __lasx_xvld(values, 32);

        in1  = __lasx_xvand_v(in1, mask);
        in0  = __lasx_xvand_v(in0, mask);
        tmp0 = __lasx_xvilvl_b(in0, zero);
        tmp1 = __lasx_xvilvh_b(in0, zero);
        t0   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t1   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_b(in1, zero);
        tmp1 = __lasx_xvilvh_b(in1, zero);
        t2   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t3   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t3);
        tmp1 = __lasx_xvilvh_h(zero, t3);
        t6   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t7   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t2);
        tmp1 = __lasx_xvilvh_h(zero, t2);
        t4   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t5   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t1);
        tmp1 = __lasx_xvilvh_h(zero, t1);
        t2   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t3   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        tmp0 = __lasx_xvilvl_h(zero, t0);
        tmp1 = __lasx_xvilvh_h(zero, t0);
        t0   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t1   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);

        LASX_TRANSPOSE8x8_W(t0, t1, t2, t3, t4, t5, t6, t7,
                            t0, t1, t2, t3, t4, t5, t6, t7);

        t0  = __lasx_xvsrli_h(t0, 12);
        t1  = __lasx_xvsrli_h(t1, 8);
        t2  = __lasx_xvsrli_h(t2, 4);
        t4  = __lasx_xvsrli_h(t4, 12);
        t5  = __lasx_xvsrli_h(t5, 8);
        t6  = __lasx_xvsrli_h(t6, 4);

        t3  = __lasx_xvor_v(t3, t2);
        t3  = __lasx_xvor_v(t3, t1);
        t3  = __lasx_xvor_v(t3, t0);
        t7  = __lasx_xvor_v(t7, t6);
        t7  = __lasx_xvor_v(t7, t5);
        t7  = __lasx_xvor_v(t7, t4);

        tmp0 = __lasx_xvilvl_w(t7, t3);
        tmp1 = __lasx_xvilvh_w(t7, t3);
        t0   = __lasx_xvpermi_q(tmp0, tmp1, 0x02);
        t1   = __lasx_xvpermi_q(tmp0, tmp1, 0x13);
        t0   = __lasx_xvpickev_h(t1, t0);
        t0   = __lasx_xvpermi_d(t0, 0xd8);
        __lasx_xvst(t0, dest, 0);
        values += 16, width -= 16, dest += 16;
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

static const pixman_fast_path_t lasx_fast_paths[] =
{
    PIXMAN_STD_FAST_PATH (OVER, solid, a8, a8r8g8b8, lasx_composite_over_n_8_8888),
    PIXMAN_STD_FAST_PATH (OVER, solid, a8, x8r8g8b8, lasx_composite_over_n_8_8888),
    PIXMAN_STD_FAST_PATH (OVER, solid, a8, a8b8g8r8, lasx_composite_over_n_8_8888),
    PIXMAN_STD_FAST_PATH (OVER, solid, a8, x8b8g8r8, lasx_composite_over_n_8_8888),
    PIXMAN_STD_FAST_PATH_CA (OVER, solid, a8r8g8b8, r5g6b5, lasx_composite_over_n_8888_0565_ca),
    PIXMAN_STD_FAST_PATH_CA (OVER, solid, a8b8g8r8, b5g6r5, lasx_composite_over_n_8888_0565_ca),
    PIXMAN_STD_FAST_PATH (OVER, x8r8g8b8, solid, a8r8g8b8, lasx_composite_over_x888_n_8888),
    PIXMAN_STD_FAST_PATH (OVER, x8r8g8b8, solid, x8r8g8b8, lasx_composite_over_x888_n_8888),
    PIXMAN_STD_FAST_PATH (OVER, x8b8g8r8, solid, a8b8g8r8, lasx_composite_over_x888_n_8888),
    PIXMAN_STD_FAST_PATH (OVER, x8b8g8r8, solid, x8b8g8r8, lasx_composite_over_x888_n_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8r8g8b8, solid, a8r8g8b8, lasx_composite_over_8888_n_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8r8g8b8, solid, x8r8g8b8, lasx_composite_over_8888_n_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8b8g8r8, solid, a8b8g8r8, lasx_composite_over_8888_n_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8b8g8r8, solid, x8b8g8r8, lasx_composite_over_8888_n_8888),
    PIXMAN_STD_FAST_PATH (OVER, x8r8g8b8, a8, x8r8g8b8, lasx_composite_over_x888_8_8888),
    PIXMAN_STD_FAST_PATH (OVER, x8r8g8b8, a8, a8r8g8b8, lasx_composite_over_x888_8_8888),
    PIXMAN_STD_FAST_PATH (OVER, x8b8g8r8, a8, x8b8g8r8, lasx_composite_over_x888_8_8888),
    PIXMAN_STD_FAST_PATH (OVER, x8b8g8r8, a8, a8b8g8r8, lasx_composite_over_x888_8_8888),
    PIXMAN_STD_FAST_PATH (OVER, solid, a8, r5g6b5, lasx_composite_over_n_8_0565),
    PIXMAN_STD_FAST_PATH (OVER, solid, a8, b5g6r5, lasx_composite_over_n_8_0565),
    PIXMAN_STD_FAST_PATH (SRC, x8r8g8b8, null, a8r8g8b8, lasx_composite_src_x888_8888),
    PIXMAN_STD_FAST_PATH (SRC, x8b8g8r8, null, a8b8g8r8, lasx_composite_src_x888_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8r8g8b8, null, r5g6b5, lasx_composite_over_8888_0565),
    PIXMAN_STD_FAST_PATH (OVER, a8b8g8r8, null, b5g6r5, lasx_composite_over_8888_0565),
    PIXMAN_STD_FAST_PATH (OVER, solid, null, r5g6b5, lasx_composite_over_n_0565),
    PIXMAN_STD_FAST_PATH (OVER, solid, null, b5g6r5, lasx_composite_over_n_0565),
    PIXMAN_STD_FAST_PATH (OVER, a8r8g8b8, null, a8r8g8b8, lasx_composite_over_8888_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8r8g8b8, null, x8r8g8b8, lasx_composite_over_8888_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8b8g8r8, null, a8b8g8r8, lasx_composite_over_8888_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8b8g8r8, null, x8b8g8r8, lasx_composite_over_8888_8888),
    PIXMAN_STD_FAST_PATH (OVER, x8r8g8b8, null, x8r8g8b8, lasx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (OVER, x8b8g8r8, null, x8b8g8r8, lasx_composite_copy_area),
    PIXMAN_STD_FAST_PATH_CA (OVER, solid, a8r8g8b8, a8r8g8b8, lasx_composite_over_n_8888_8888_ca),
    PIXMAN_STD_FAST_PATH_CA (OVER, solid, a8r8g8b8, x8r8g8b8, lasx_composite_over_n_8888_8888_ca),
    PIXMAN_STD_FAST_PATH_CA (OVER, solid, a8b8g8r8, a8b8g8r8, lasx_composite_over_n_8888_8888_ca),
    PIXMAN_STD_FAST_PATH_CA (OVER, solid, a8b8g8r8, x8b8g8r8, lasx_composite_over_n_8888_8888_ca),
    PIXMAN_STD_FAST_PATH (OVER_REVERSE, solid, null, a8r8g8b8, lasx_composite_over_reverse_n_8888),
    PIXMAN_STD_FAST_PATH (OVER_REVERSE, solid, null, a8b8g8r8, lasx_composite_over_reverse_n_8888),
    PIXMAN_STD_FAST_PATH (ADD, a8, null, a8, lasx_composite_add_8_8),
    PIXMAN_STD_FAST_PATH (ADD, solid, a8, a8, lasx_composite_add_n_8_8),
    PIXMAN_STD_FAST_PATH (ADD, solid, null, a8, lasx_composite_add_n_8),
    PIXMAN_STD_FAST_PATH (ADD, solid, null, x8r8g8b8, lasx_composite_add_n_8888),
    PIXMAN_STD_FAST_PATH (ADD, solid, null, a8r8g8b8, lasx_composite_add_n_8888),
    PIXMAN_STD_FAST_PATH (ADD, solid, null, x8b8g8r8, lasx_composite_add_n_8888),
    PIXMAN_STD_FAST_PATH (ADD, solid, null, a8b8g8r8, lasx_composite_add_n_8888),
    PIXMAN_STD_FAST_PATH (ADD, a8r8g8b8, null, a8r8g8b8, lasx_composite_add_8888_8888),
    PIXMAN_STD_FAST_PATH (ADD, a8b8g8r8, null, a8b8g8r8, lasx_composite_add_8888_8888),
    PIXMAN_STD_FAST_PATH (SRC, a8r8g8b8, null, a8r8g8b8, lasx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, a8b8g8r8, null, a8b8g8r8, lasx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, a8r8g8b8, null, x8r8g8b8, lasx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, a8b8g8r8, null, x8b8g8r8, lasx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, x8r8g8b8, null, x8r8g8b8, lasx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, x8b8g8r8, null, x8b8g8r8, lasx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, b8g8r8a8, null, b8g8r8x8, lasx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, b8g8r8a8, null, b8g8r8a8, lasx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, b8g8r8x8, null, b8g8r8x8, lasx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, r5g6b5,   null, r5g6b5,   lasx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, b5g6r5,   null, b5g6r5,   lasx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, a8,       null, a8,       lasx_composite_copy_area),
    PIXMAN_STD_FAST_PATH (SRC, a8r8g8b8, null, r5g6b5, lasx_composite_src_x888_0565),
    PIXMAN_STD_FAST_PATH (SRC, a8b8g8r8, null, b5g6r5, lasx_composite_src_x888_0565),
    PIXMAN_STD_FAST_PATH (SRC, x8r8g8b8, null, r5g6b5, lasx_composite_src_x888_0565),
    PIXMAN_STD_FAST_PATH (SRC, x8b8g8r8, null, b5g6r5, lasx_composite_src_x888_0565),
    PIXMAN_STD_FAST_PATH (IN, solid, a8, a8, lasx_composite_in_n_8_8),
    PIXMAN_STD_FAST_PATH (IN, a8, null, a8, lasx_composite_in_8_8),

    { PIXMAN_OP_NONE },
};

#define IMAGE_FLAGS                                                     \
    (FAST_PATH_STANDARD_FLAGS | FAST_PATH_ID_TRANSFORM |                \
     FAST_PATH_BITS_IMAGE | FAST_PATH_SAMPLES_COVER_CLIP_NEAREST)
static const pixman_iter_info_t lasx_iters[] =
{
    {
      PIXMAN_x8r8g8b8, IMAGE_FLAGS, ITER_NARROW,
      _pixman_iter_init_bits_stride, lasx_fetch_x8r8g8b8, NULL
    },
    {
      PIXMAN_r5g6b5, IMAGE_FLAGS, ITER_NARROW,
      _pixman_iter_init_bits_stride, lasx_fetch_r5g6b5, NULL
    },
    {
      PIXMAN_a8, IMAGE_FLAGS, ITER_NARROW,
      _pixman_iter_init_bits_stride, lasx_fetch_a8, NULL
    },
    { PIXMAN_null },
};

pixman_implementation_t *
_pixman_implementation_create_lasx (pixman_implementation_t *fallback)
{
    pixman_implementation_t *imp =
        _pixman_implementation_create (fallback, lasx_fast_paths);

    /* LoongArch LASX constants */
    mask_565_r  = create_mask_1x32_256 (0x00f80000);
    mask_565_g1 = create_mask_1x32_256 (0x00070000);
    mask_565_g2 = create_mask_1x32_256 (0x000000e0);
    mask_565_b  = create_mask_1x32_256 (0x0000001f);
    mask_red   = create_mask_1x32_256 (0x00f80000);
    mask_green = create_mask_1x32_256 (0x0000fc00);
    mask_blue  = create_mask_1x32_256 (0x000000f8);
    mask_565_fix_rb = create_mask_1x32_256 (0x00e000e0);
    mask_565_fix_g = create_mask_1x32_256  (0x0000c000);
    mask_0080 = create_mask_16_256 (0x0080);
    mask_00ff = create_mask_16_256 (0x00ff);
    mask_0101 = create_mask_16_256 (0x0101);
    mask_ffff = create_mask_16_256 (0xffff);
    mask_ff000000 = create_mask_1x32_256 (0xff000000);
    mask_alpha = create_mask_1x64_256 (0x00ff000000000000);
    mask_565_rb = create_mask_1x32_256 (0x00f800f8);
    mask_565_pack_multiplier = create_mask_1x32_256 (0x20000004);

    /* Set up function pointers */
    imp->combine_32[PIXMAN_OP_SRC] = lasx_combine_src_u;
    imp->combine_32[PIXMAN_OP_OVER] = lasx_combine_over_u;
    imp->combine_32[PIXMAN_OP_OVER_REVERSE] = lasx_combine_over_reverse_u;
    imp->combine_32[PIXMAN_OP_OUT] = lasx_combine_out_u;
    imp->combine_32[PIXMAN_OP_OUT_REVERSE] = lasx_combine_out_reverse_u;
    imp->combine_32[PIXMAN_OP_ADD] = lasx_combine_add_u;
    imp->combine_32[PIXMAN_OP_DISJOINT_SRC] = lasx_combine_src_u;
    imp->combine_32[PIXMAN_OP_CONJOINT_SRC] = lasx_combine_src_u;
    imp->combine_32[PIXMAN_OP_MULTIPLY] = lasx_combine_multiply_u;
    imp->combine_32_ca[PIXMAN_OP_SRC] = lasx_combine_src_ca;
    imp->combine_32_ca[PIXMAN_OP_OVER] = lasx_combine_over_ca;
    imp->combine_32_ca[PIXMAN_OP_OUT_REVERSE] = lasx_combine_out_reverse_ca;

    imp->blt = lasx_blt;
    imp->fill = lasx_fill;
    imp->iter_info = lasx_iters;

    return imp;
}

void setup_accessors_lasx (bits_image_t *image)
{
    if (image->format == PIXMAN_a8) { // 8 bits
        image->fetch_scanline_32 = lasx_fetch_scanline_a8;
        image->store_scanline_32 = lasx_store_scanline_a8;
    } else if (image->format == PIXMAN_a2r2g2b2) {
        image->fetch_scanline_32 = lasx_fetch_scanline_a2r2g2b2;
        image->store_scanline_32 = lasx_store_scanline_a2r2g2b2;
    } else if (image->format == PIXMAN_a1r5g5b5) { // 16 bits
        image->fetch_scanline_32 = lasx_fetch_scanline_a1r5g5b5;
        image->store_scanline_32 = lasx_store_scanline_a1r5g5b5;
    } else if (image->format == PIXMAN_a4r4g4b4) {
        image->fetch_scanline_32 = lasx_fetch_scanline_a4r4g4b4;
        image->store_scanline_32 = lasx_store_scanline_a4r4g4b4;
    }
}
