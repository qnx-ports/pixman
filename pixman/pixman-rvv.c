

/*
 * Copyright © 2000 Keith Packard, member of The XFree86 Project, Inc.
 *             2005 Lars Knoll & Zack Rusin, Trolltech
 *             2024 Bernard Gingold, Samsung Electronics
 * Permission to use, copy, modify, distribute, and sell this software and its
 * documentation for any purpose is hereby granted without fee, provided that
 * the above copyright notice appear in all copies and that both that
 * copyright notice and this permission notice appear in supporting
 * documentation, and that the name of Keith Packard not be used in
 * advertising or publicity pertaining to distribution of the software without
 * specific, written prior permission.  Keith Packard makes no
 * representations about the suitability of this software for any purpose.  It
 * is provided "as is" without express or implied warranty.
 *
 * THE COPYRIGHT HOLDERS DISCLAIM ALL WARRANTIES WITH REGARD TO THIS
 * SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS, IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
 * AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
 * OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOFTWARE.
 */

#ifdef HAVE_CONFIG_H
#include <pixman-config.h>
#endif

#include "pixman-private.h"
#include <stdbool.h>
#include <stddef.h>

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pixman-combine-float.h"

#ifdef USE_RVV

#include "pixman-rvv-blendings-float.h"
#include <riscv_vector.h>

static const pixman_fast_path_t rvv_fast_paths[] = {
    {PIXMAN_OP_NONE},
};

pixman_implementation_t *
_pixman_implementation_create_rvv(pixman_implementation_t *fallback)
{
    pixman_implementation_t *imp =
        _pixman_implementation_create(fallback, rvv_fast_paths);

    imp->combine_float[PIXMAN_OP_CLEAR] = rvv_combine_clear_u_float;
    imp->combine_float[PIXMAN_OP_SRC] = rvv_combine_src_u_float;
    imp->combine_float[PIXMAN_OP_DST] = rvv_combine_dst_u_float;
    imp->combine_float[PIXMAN_OP_OVER] = rvv_combine_over_u_float;
    imp->combine_float[PIXMAN_OP_OVER_REVERSE] =
        rvv_combine_over_reverse_u_float;
    imp->combine_float[PIXMAN_OP_IN] = rvv_combine_in_u_float;
    imp->combine_float[PIXMAN_OP_IN_REVERSE] = rvv_combine_in_reverse_u_float;
    imp->combine_float[PIXMAN_OP_OUT] = rvv_combine_out_u_float;
    imp->combine_float[PIXMAN_OP_OUT_REVERSE] = rvv_combine_out_reverse_u_float;
    imp->combine_float[PIXMAN_OP_ATOP] = rvv_combine_atop_u_float;
    imp->combine_float[PIXMAN_OP_ATOP_REVERSE] =
        rvv_combine_atop_reverse_u_float;
    imp->combine_float[PIXMAN_OP_XOR] = rvv_combine_xor_u_float;
    imp->combine_float[PIXMAN_OP_ADD] = rvv_combine_add_u_float;
    imp->combine_float[PIXMAN_OP_SATURATE] = rvv_combine_saturate_u_float;

    /* Disjoint, unified */
    imp->combine_float[PIXMAN_OP_DISJOINT_CLEAR] =
        rvv_combine_disjoint_clear_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_SRC] =
        rvv_combine_disjoint_src_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_DST] =
        rvv_combine_disjoint_dst_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_OVER] =
        rvv_combine_disjoint_over_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_OVER_REVERSE] =
        rvv_combine_disjoint_over_reverse_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_IN] = rvv_combine_disjoint_in_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_IN_REVERSE] =
        rvv_combine_disjoint_in_reverse_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_OUT] =
        rvv_combine_disjoint_out_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_OUT_REVERSE] =
        rvv_combine_disjoint_out_reverse_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_ATOP] =
        rvv_combine_disjoint_atop_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_ATOP_REVERSE] =
        rvv_combine_disjoint_atop_reverse_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_XOR] =
        rvv_combine_disjoint_xor_u_float;

    /* Conjoint, unified */
    imp->combine_float[PIXMAN_OP_CONJOINT_CLEAR] =
        rvv_combine_conjoint_clear_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_SRC] =
        rvv_combine_conjoint_src_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_DST] =
        rvv_combine_conjoint_dst_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_OVER] =
        rvv_combine_conjoint_over_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_OVER_REVERSE] =
        rvv_combine_conjoint_over_reverse_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_IN] = rvv_combine_conjoint_in_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_IN_REVERSE] =
        rvv_combine_conjoint_in_reverse_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_OUT] =
        rvv_combine_conjoint_out_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_OUT_REVERSE] =
        rvv_combine_conjoint_out_reverse_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_ATOP] =
        rvv_combine_conjoint_atop_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_ATOP_REVERSE] =
        rvv_combine_conjoint_atop_reverse_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_XOR] =
        rvv_combine_conjoint_xor_u_float;

    /* PDF operators, unified */
    imp->combine_float[PIXMAN_OP_MULTIPLY] = rvv_combine_multiply_u_float;
    imp->combine_float[PIXMAN_OP_SCREEN] = rvv_combine_screen_u_float;
    imp->combine_float[PIXMAN_OP_OVERLAY] = rvv_combine_overlay_u_float;
    imp->combine_float[PIXMAN_OP_DARKEN] = rvv_combine_darken_u_float;
    imp->combine_float[PIXMAN_OP_LIGHTEN] = rvv_combine_lighten_u_float;

    imp->combine_float[PIXMAN_OP_HARD_LIGHT] = rvv_combine_hard_light_u_float;
    imp->combine_float[PIXMAN_OP_SOFT_LIGHT] = rvv_combine_soft_light_u_float;
    imp->combine_float[PIXMAN_OP_DIFFERENCE] = rvv_combine_difference_u_float;
    imp->combine_float[PIXMAN_OP_EXCLUSION] = rvv_combine_exclusion_u_float;

    /* Component alpha combiners */
    imp->combine_float_ca[PIXMAN_OP_CLEAR] = rvv_combine_clear_ca_float;
    imp->combine_float_ca[PIXMAN_OP_SRC] = rvv_combine_src_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DST] = rvv_combine_dst_ca_float;
    imp->combine_float_ca[PIXMAN_OP_OVER] = rvv_combine_over_ca_float;
    imp->combine_float_ca[PIXMAN_OP_OVER_REVERSE] =
        rvv_combine_over_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_IN] = rvv_combine_in_ca_float;
    imp->combine_float_ca[PIXMAN_OP_IN_REVERSE] =
        rvv_combine_in_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_OUT] = rvv_combine_out_ca_float;
    imp->combine_float_ca[PIXMAN_OP_OUT_REVERSE] =
        rvv_combine_out_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_ATOP] = rvv_combine_atop_ca_float;
    imp->combine_float_ca[PIXMAN_OP_ATOP_REVERSE] =
        rvv_combine_atop_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_XOR] = rvv_combine_xor_ca_float;
    imp->combine_float_ca[PIXMAN_OP_ADD] = rvv_combine_add_ca_float;
    imp->combine_float_ca[PIXMAN_OP_SATURATE] = rvv_combine_saturate_ca_float;

    /* Disjoint CA */
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_CLEAR] =
        rvv_combine_disjoint_clear_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_SRC] =
        rvv_combine_disjoint_src_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_DST] =
        rvv_combine_disjoint_dst_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_OVER] =
        rvv_combine_disjoint_over_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_OVER_REVERSE] =
        rvv_combine_disjoint_over_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_IN] =
        rvv_combine_disjoint_in_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_IN_REVERSE] =
        rvv_combine_disjoint_in_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_OUT] =
        rvv_combine_disjoint_out_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_OUT_REVERSE] =
        rvv_combine_disjoint_out_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_ATOP] =
        rvv_combine_disjoint_atop_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_ATOP_REVERSE] =
        rvv_combine_disjoint_atop_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_XOR] =
        rvv_combine_disjoint_xor_ca_float;

    /* Conjoint CA */
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_CLEAR] =
        rvv_combine_conjoint_clear_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_SRC] =
        rvv_combine_conjoint_src_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_DST] =
        rvv_combine_conjoint_dst_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_OVER] =
        rvv_combine_conjoint_over_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_OVER_REVERSE] =
        rvv_combine_conjoint_over_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_IN] =
        rvv_combine_conjoint_in_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_IN_REVERSE] =
        rvv_combine_conjoint_in_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_OUT] =
        rvv_combine_conjoint_out_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_OUT_REVERSE] =
        rvv_combine_conjoint_out_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_ATOP] =
        rvv_combine_conjoint_atop_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_ATOP_REVERSE] =
        rvv_combine_conjoint_atop_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_XOR] =
        rvv_combine_conjoint_xor_ca_float;

    /* PDF operators CA */
    imp->combine_float_ca[PIXMAN_OP_MULTIPLY] = rvv_combine_multiply_ca_float;
    imp->combine_float_ca[PIXMAN_OP_SCREEN] = rvv_combine_screen_ca_float;
    imp->combine_float_ca[PIXMAN_OP_OVERLAY] = rvv_combine_overlay_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DARKEN] = rvv_combine_darken_ca_float;
    imp->combine_float_ca[PIXMAN_OP_LIGHTEN] = rvv_combine_lighten_ca_float;

    imp->combine_float_ca[PIXMAN_OP_HARD_LIGHT] =
        rvv_combine_hard_light_ca_float;
    imp->combine_float_ca[PIXMAN_OP_SOFT_LIGHT] =
        rvv_combine_soft_light_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DIFFERENCE] =
        rvv_combine_difference_ca_float;
    imp->combine_float_ca[PIXMAN_OP_EXCLUSION] = rvv_combine_exclusion_ca_float;

    /* It is not clear that these make sense, so make them noops for now */
    imp->combine_float_ca[PIXMAN_OP_HSL_HUE] = rvv_combine_dst_u_float;
    imp->combine_float_ca[PIXMAN_OP_HSL_SATURATION] = rvv_combine_dst_u_float;
    imp->combine_float_ca[PIXMAN_OP_HSL_COLOR] = rvv_combine_dst_u_float;
    imp->combine_float_ca[PIXMAN_OP_HSL_LUMINOSITY] = rvv_combine_dst_u_float;

    return imp;
}

#endif // USE_RVV

pixman_implementation_t *
_pixman_riscv_get_implementations(pixman_implementation_t *imp)
{
#ifdef USE_RVV
    if (!_pixman_disabled("have_rvv"))
    {
        imp = _pixman_implementation_create_rvv(imp);
    }
#endif

    return imp;
}
