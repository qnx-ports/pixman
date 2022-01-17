#!/bin/sh

set -e -x

TEST_LOG=build/meson-logs/testlog.txt
CROSS_FILE=/cross_file-${CROSS}.txt

if [ -n "$CROSS" ]; then
    CROSS_XFAIL=.gitlab-ci/cross-xfail-"$CROSS"
    if [ -s "$CROSS_XFAIL" ]; then
        sed -i \
            -e '/\[properties\]/a\' \
            -e "xfail = '$(tr '\n' , < $CROSS_XFAIL)'" \
            "$CROSS_FILE"
    fi
fi

meson setup build \
    ${CROSS+--cross "$CROSS_FILE"} \
    ${UBSAN+-D b_sanitize=undefined}

meson configure build

ninja -C build

meson test -C build ${CROSS+-t 20}

# use the native runners to exercise the less-optimized paths as well
if [ -z "$CROSS" -o "$CROSS" = i386 ]; then
    mv ${TEST_LOG} ${TEST_LOG}.normal
    PIXMAN_DISABLE="sss3" meson test -C build
    mv ${TEST_LOG} ${TEST_LOG}.no-ssse3.txt
    PIXMAN_DISABLE="sss3 sse2" meson test -C build
    mv ${TEST_LOG} ${TEST_LOG}.no-sse2.txt
    PIXMAN_DISABLE="sss3 sse2 mmx" meson test -C build
    mv ${TEST_LOG} ${TEST_LOG}.no-mmx.txt
    PIXMAN_DISABLE="sss3 sse2 mmx fast" meson test -C build
    mv ${TEST_LOG} ${TEST_LOG}.no-fast.txt
fi

# eventually this should be -D c_args=-fno-sanitize-recover
if [ -n "$UBSAN" ]; then
    grep runtime.error build/meson-logs/testlog.txt* && exit 1
fi
