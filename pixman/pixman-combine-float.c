/* -*- Mode: c; c-basic-offset: 4; tab-width: 8; indent-tabs-mode: t; -*- */
/*
 * Copyright © 2010, 2012 Soren Sandmann Pedersen
 * Copyright © 2010, 2012 Red Hat, Inc.
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
 * Author: Soren Sandmann Pedersen (sandmann@cs.au.dk)
 */

#ifdef HAVE_CONFIG_H
#include <pixman-config.h>
#endif

#include <math.h>
#include <string.h>
#include <float.h>

#include "pixman-private.h"
#include "pixman-combine-float.h"

void
_pixman_setup_combiner_functions_float (pixman_implementation_t *imp)
{
    /* Unified alpha */
    imp->combine_float[PIXMAN_OP_CLEAR] = combine_clear_u_float;
    imp->combine_float[PIXMAN_OP_SRC] = combine_src_u_float;
    imp->combine_float[PIXMAN_OP_DST] = combine_dst_u_float;
    imp->combine_float[PIXMAN_OP_OVER] = combine_over_u_float;
    imp->combine_float[PIXMAN_OP_OVER_REVERSE] = combine_over_reverse_u_float;
    imp->combine_float[PIXMAN_OP_IN] = combine_in_u_float;
    imp->combine_float[PIXMAN_OP_IN_REVERSE] = combine_in_reverse_u_float;
    imp->combine_float[PIXMAN_OP_OUT] = combine_out_u_float;
    imp->combine_float[PIXMAN_OP_OUT_REVERSE] = combine_out_reverse_u_float;
    imp->combine_float[PIXMAN_OP_ATOP] = combine_atop_u_float;
    imp->combine_float[PIXMAN_OP_ATOP_REVERSE] = combine_atop_reverse_u_float;
    imp->combine_float[PIXMAN_OP_XOR] = combine_xor_u_float;
    imp->combine_float[PIXMAN_OP_ADD] = combine_add_u_float;
    imp->combine_float[PIXMAN_OP_SATURATE] = combine_saturate_u_float;

    /* Disjoint, unified */
    imp->combine_float[PIXMAN_OP_DISJOINT_CLEAR] = combine_disjoint_clear_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_SRC] = combine_disjoint_src_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_DST] = combine_disjoint_dst_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_OVER] = combine_disjoint_over_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_OVER_REVERSE] = combine_disjoint_over_reverse_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_IN] = combine_disjoint_in_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_IN_REVERSE] = combine_disjoint_in_reverse_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_OUT] = combine_disjoint_out_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_OUT_REVERSE] = combine_disjoint_out_reverse_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_ATOP] = combine_disjoint_atop_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_ATOP_REVERSE] = combine_disjoint_atop_reverse_u_float;
    imp->combine_float[PIXMAN_OP_DISJOINT_XOR] = combine_disjoint_xor_u_float;

    /* Conjoint, unified */
    imp->combine_float[PIXMAN_OP_CONJOINT_CLEAR] = combine_conjoint_clear_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_SRC] = combine_conjoint_src_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_DST] = combine_conjoint_dst_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_OVER] = combine_conjoint_over_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_OVER_REVERSE] = combine_conjoint_over_reverse_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_IN] = combine_conjoint_in_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_IN_REVERSE] = combine_conjoint_in_reverse_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_OUT] = combine_conjoint_out_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_OUT_REVERSE] = combine_conjoint_out_reverse_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_ATOP] = combine_conjoint_atop_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_ATOP_REVERSE] = combine_conjoint_atop_reverse_u_float;
    imp->combine_float[PIXMAN_OP_CONJOINT_XOR] = combine_conjoint_xor_u_float;

    /* PDF operators, unified */
    imp->combine_float[PIXMAN_OP_MULTIPLY] = combine_multiply_u_float;
    imp->combine_float[PIXMAN_OP_SCREEN] = combine_screen_u_float;
    imp->combine_float[PIXMAN_OP_OVERLAY] = combine_overlay_u_float;
    imp->combine_float[PIXMAN_OP_DARKEN] = combine_darken_u_float;
    imp->combine_float[PIXMAN_OP_LIGHTEN] = combine_lighten_u_float;
    imp->combine_float[PIXMAN_OP_COLOR_DODGE] = combine_color_dodge_u_float;
    imp->combine_float[PIXMAN_OP_COLOR_BURN] = combine_color_burn_u_float;
    imp->combine_float[PIXMAN_OP_HARD_LIGHT] = combine_hard_light_u_float;
    imp->combine_float[PIXMAN_OP_SOFT_LIGHT] = combine_soft_light_u_float;
    imp->combine_float[PIXMAN_OP_DIFFERENCE] = combine_difference_u_float;
    imp->combine_float[PIXMAN_OP_EXCLUSION] = combine_exclusion_u_float;

    imp->combine_float[PIXMAN_OP_HSL_HUE] = combine_hsl_hue_u_float;
    imp->combine_float[PIXMAN_OP_HSL_SATURATION] = combine_hsl_saturation_u_float;
    imp->combine_float[PIXMAN_OP_HSL_COLOR] = combine_hsl_color_u_float;
    imp->combine_float[PIXMAN_OP_HSL_LUMINOSITY] = combine_hsl_luminosity_u_float;

    /* Component alpha combiners */
    imp->combine_float_ca[PIXMAN_OP_CLEAR] = combine_clear_ca_float;
    imp->combine_float_ca[PIXMAN_OP_SRC] = combine_src_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DST] = combine_dst_ca_float;
    imp->combine_float_ca[PIXMAN_OP_OVER] = combine_over_ca_float;
    imp->combine_float_ca[PIXMAN_OP_OVER_REVERSE] = combine_over_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_IN] = combine_in_ca_float;
    imp->combine_float_ca[PIXMAN_OP_IN_REVERSE] = combine_in_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_OUT] = combine_out_ca_float;
    imp->combine_float_ca[PIXMAN_OP_OUT_REVERSE] = combine_out_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_ATOP] = combine_atop_ca_float;
    imp->combine_float_ca[PIXMAN_OP_ATOP_REVERSE] = combine_atop_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_XOR] = combine_xor_ca_float;
    imp->combine_float_ca[PIXMAN_OP_ADD] = combine_add_ca_float;
    imp->combine_float_ca[PIXMAN_OP_SATURATE] = combine_saturate_ca_float;

    /* Disjoint CA */
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_CLEAR] = combine_disjoint_clear_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_SRC] = combine_disjoint_src_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_DST] = combine_disjoint_dst_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_OVER] = combine_disjoint_over_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_OVER_REVERSE] = combine_disjoint_over_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_IN] = combine_disjoint_in_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_IN_REVERSE] = combine_disjoint_in_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_OUT] = combine_disjoint_out_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_OUT_REVERSE] = combine_disjoint_out_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_ATOP] = combine_disjoint_atop_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_ATOP_REVERSE] = combine_disjoint_atop_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DISJOINT_XOR] = combine_disjoint_xor_ca_float;

    /* Conjoint CA */
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_CLEAR] = combine_conjoint_clear_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_SRC] = combine_conjoint_src_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_DST] = combine_conjoint_dst_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_OVER] = combine_conjoint_over_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_OVER_REVERSE] = combine_conjoint_over_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_IN] = combine_conjoint_in_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_IN_REVERSE] = combine_conjoint_in_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_OUT] = combine_conjoint_out_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_OUT_REVERSE] = combine_conjoint_out_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_ATOP] = combine_conjoint_atop_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_ATOP_REVERSE] = combine_conjoint_atop_reverse_ca_float;
    imp->combine_float_ca[PIXMAN_OP_CONJOINT_XOR] = combine_conjoint_xor_ca_float;

    /* PDF operators CA */
    imp->combine_float_ca[PIXMAN_OP_MULTIPLY] = combine_multiply_ca_float;
    imp->combine_float_ca[PIXMAN_OP_SCREEN] = combine_screen_ca_float;
    imp->combine_float_ca[PIXMAN_OP_OVERLAY] = combine_overlay_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DARKEN] = combine_darken_ca_float;
    imp->combine_float_ca[PIXMAN_OP_LIGHTEN] = combine_lighten_ca_float;
    imp->combine_float_ca[PIXMAN_OP_COLOR_DODGE] = combine_color_dodge_ca_float;
    imp->combine_float_ca[PIXMAN_OP_COLOR_BURN] = combine_color_burn_ca_float;
    imp->combine_float_ca[PIXMAN_OP_HARD_LIGHT] = combine_hard_light_ca_float;
    imp->combine_float_ca[PIXMAN_OP_SOFT_LIGHT] = combine_soft_light_ca_float;
    imp->combine_float_ca[PIXMAN_OP_DIFFERENCE] = combine_difference_ca_float;
    imp->combine_float_ca[PIXMAN_OP_EXCLUSION] = combine_exclusion_ca_float;

    /* It is not clear that these make sense, so make them noops for now */
    imp->combine_float_ca[PIXMAN_OP_HSL_HUE] = combine_dst_u_float;
    imp->combine_float_ca[PIXMAN_OP_HSL_SATURATION] = combine_dst_u_float;
    imp->combine_float_ca[PIXMAN_OP_HSL_COLOR] = combine_dst_u_float;
    imp->combine_float_ca[PIXMAN_OP_HSL_LUMINOSITY] = combine_dst_u_float;
}
