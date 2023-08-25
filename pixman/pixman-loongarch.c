/*
 * Copyright (c) 2023 Loongson Technology Corporation Limited
 * Contributed by Lu Wang<wanglu@loongson.cn>
 *                Ding Song<songding@loongson.cn>
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

#if defined(USE_LOONGARCH_LSX) || defined(USE_LOONGARCH_LASX)
#include <string.h>
#include <stdlib.h>
#include <sys/auxv.h>

#ifdef USE_LOONGARCH_LSX
static int have_lsx = 0;
#endif
#ifdef USE_LOONGARCH_LASX
static int have_lasx = 0;
#endif

static uint64_t detect_cpu_features (void)
{
    uint64_t hwcap = 0;
    hwcap = getauxval(AT_HWCAP);

    return hwcap;
}

static pixman_bool_t
have_feature (uint64_t feature)
{
    static pixman_bool_t initialized;
    static uint64_t features;

    if (!initialized)
    {
        features = detect_cpu_features();
        initialized = TRUE;
    }

    return (features & feature) == feature;
}

#endif

pixman_implementation_t *
_pixman_loongarch_get_implementations (pixman_implementation_t *imp)
{
#ifdef USE_LOONGARCH_LSX
    if (!_pixman_disabled ("loongarch-lsx") && have_feature (HWCAP_LOONGARCH_LSX))
    {
        imp = _pixman_implementation_create_lsx (imp);
        have_lsx = 1;
    }
#endif
#ifdef USE_LOONGARCH_LASX
    if (!_pixman_disabled ("loongarch-lasx") && have_feature (HWCAP_LOONGARCH_LASX))
    {
        imp = _pixman_implementation_create_lasx (imp);
        have_lasx = 1;
    }
#endif
    return imp;
}

void setup_loongarch_accessors (bits_image_t *image)
{
#ifdef USE_LOONGARCH_LSX
    if (have_lsx)
        setup_accessors_lsx(image);
#endif
#ifdef USE_LOONGARCH_LASX
    if (have_lasx)
        setup_accessors_lasx(image);
#endif
}
