#!/bin/bash -e

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

echo "Running NNFusion install_dependency.sh"
DEB_PACKAGES="build-essential cmake git curl zlib1g zlib1g-dev libtinfo-dev unzip \
autoconf automake libtool ca-certificates gdb sqlite3 libsqlite3-dev libcurl4-openssl-dev \
libprotobuf-dev protobuf-compiler libgflags-dev libgtest-dev"

ubuntu_codename=$(. /etc/os-release;echo $UBUNTU_CODENAME)

if [[ $ubuntu_codename != "focal" ]]; then
	DEB_PACKAGES="${DEB_PACKAGES} clang-3.9 clang-format-3.9"
fi

if [[ "$(whoami)" != "root" ]]; then
	SUDO=sudo
fi

if ! dpkg -L $DEB_PACKAGES >/dev/null 2>&1; then
	#Thirdparty deb for ubuntu 18.04(bionic)
	$SUDO sh -c "apt update && apt install -y --no-install-recommends software-properties-common apt-transport-https ca-certificates gnupg wget"
	$SUDO sh -c "wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null"
	$SUDO sh -c "apt-add-repository 'deb https://apt.kitware.com/ubuntu/ $ubuntu_codename main'"
	$SUDO sh -c "apt update && apt install -y --no-install-recommends $DEB_PACKAGES"

	if [[ $ubuntu_codename != "focal" ]]; then
		# Install protobuf 3.6.1 from source
		$SUDO sh -c "wget https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protobuf-cpp-3.6.1.tar.gz -P /tmp"
		$SUDO sh -c "cd /tmp && tar -xf /tmp/protobuf-cpp-3.6.1.tar.gz && rm /tmp/protobuf-cpp-3.6.1.tar.gz"
		$SUDO sh -c "cd /tmp/protobuf-3.6.1/ && ./configure && make && make check && make install && ldconfig && rm -rf /tmp/protobuf-3.6.1/"
	fi
fi

if [[ $ubuntu_codename == "focal" ]]; then
	# Install clang-format-3.9
	$SUDO sh -c "cd /tmp && wget https://releases.llvm.org/3.9.0/clang+llvm-3.9.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz && tar -xf clang+llvm-3.9.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz"
	$SUDO sh -c "cp /tmp/clang+llvm-3.9.0-x86_64-linux-gnu-ubuntu-16.04/bin/clang-format /usr/bin/clang-format-3.9 && ln -s /usr/bin/clang-format-3.9 /usr/bin/clang-format"
	$SUDO sh -c "rm -rf /tmp/clang+llvm-3.9.0-x86_64-linux-gnu-ubuntu-16.04/bin/clang-format /tmp/clang+llvm-3.9.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz"
fi

echo "- Dependencies are installed in system."

if [ ! -f "/usr/lib/libgtest.a" ]; then

	# if Ubuntu 16.04, we have some dev node using ubuntu 16.04
	if [[ $ubuntu_codename == "xenial" ]]; then
		$SUDO sh -c "mkdir /usr/src/googletest && ln -s /usr/src/gtest /usr/src/googletest/googletest"
	fi

	# Compile gtest
	$SUDO sh -c "cd /usr/src/googletest/googletest/ && mkdir -p  build && cd build && cmake .. -DCMAKE_CXX_FLAGS=\"-std=c++11\" && make -j"

	if [[ $ubuntu_codename == "focal" ]]; then
		$SUDO sh -c "cp /usr/src/googletest/googletest/build/lib/libgtest*.a /usr/lib/"
	else
		$SUDO sh -c "cp /usr/src/googletest/googletest/build/libgtest*.a /usr/lib/"
	fi

	$SUDO sh -c "rm -rf /usr/src/googletest/googletest/build"
	$SUDO sh -c "mkdir /usr/local/lib/googletest"
	$SUDO sh -c "ln -s /usr/lib/libgtest.a /usr/local/lib/googletest/libgtest.a"
	$SUDO sh -c "ln -s /usr/lib/libgtest_main.a /usr/local/lib/googletest/libgtest_main.a"
fi
echo "- libgtest is installed in system."

# Install numpy
$SUDO sh -c "apt install -y python3 python3-pip"
if [[ $ubuntu_codename == "xenial" ]]; then
	$SUDO sh -c "pip3 install numpy==1.18.5"
else
	$SUDO sh -c "pip3 install numpy"
fi

echo "- Done."
