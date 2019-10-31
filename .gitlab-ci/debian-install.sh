#!/bin/sh

set -e -x

CROSS_ARCHITECTURES="i386 s390x ppc64el mips armhf arm64"
for arch in $CROSS_ARCHITECTURES; do
    dpkg --add-architecture $arch
done

apt-get install -y \
      ca-certificates \
      wget \
      unzip

sed -i -e 's/http:\/\/deb/https:\/\/deb/g' /etc/apt/sources.list
echo 'deb https://deb.debian.org/debian buster-backports main' >/etc/apt/sources.list.d/backports.list

apt-get update
apt-get install -y gcc meson pkg-config libgtk2.0-dev libpng-dev qemu-user

for arch in $CROSS_ARCHITECTURES ; do
    echo $arch
done | xargs -n1 -I@ apt-get install -y crossbuild-essential-@

for arch in $CROSS_ARCHITECTURES ; do
    cross_file="/cross_file-$arch.txt"
    /usr/share/meson/debcrossgen --arch $arch -o $cross_file
    sed -i -e '/\[properties\]/a\' -e "needs_exe_wrapper = False" "$cross_file"
done
