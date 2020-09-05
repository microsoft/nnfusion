#!/bin/bash -e

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

echo "Running NNFusion install_dependency.sh"
DEB_PACKAGES="build-essential cmake clang-3.9 clang-format-3.9 git curl zlib1g zlib1g-dev libtinfo-dev unzip \
autoconf automake libtool ca-certificates gdb sqlite3 libsqlite3-dev libcurl4-openssl-dev libprotobuf-dev \
protobuf-compiler libgflags-dev libgtest-dev"

if [[ "$(whoami)" != "root" ]]; then
    SUDO=sudo
fi

ubuntu_codename=$(. /etc/os-release;echo $UBUNTU_CODENAME)

if ! dpkg -L $DEB_PACKAGES >/dev/null 2>&1; then
	#Thirdparty deb for ubuntu 18.04(bionic)
	$SUDO sh -c "apt update && apt install -y --no-install-recommends software-properties-common apt-transport-https ca-certificates gnupg wget"
    $SUDO sh -c "wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null"
    $SUDO sh -c "apt-add-repository 'deb https://apt.kitware.com/ubuntu/ $ubuntu_codename main'"
	$SUDO sh -c "add-apt-repository ppa:maarten-fonville/protobuf -y" 
	$SUDO sh -c "apt update && apt install -y --no-install-recommends $DEB_PACKAGES"
fi
echo "- Dependencies are installed in system."
    #Install latest CMake

# if Ubuntu 16.04, we have some dev node using ubuntu 16.04
if [[ $ubuntu_codename == "xenial" ]]; then
    echo "- Ubuntu 16.04 detected. Download & install latest protobuf & gtest debs."

    # Install protobuf & gtest from bionic version
    $SUDO sh -c " \
    wget https://launchpad.net/~maarten-fonville/+archive/ubuntu/protobuf/+files/libprotobuf17_3.6.1-1~maarten0+bionic_amd64.deb -P /tmp && \
    wget https://launchpad.net/~maarten-fonville/+archive/ubuntu/protobuf/+files/libprotoc17_3.6.1-1~maarten0+bionic_amd64.deb -P /tmp && \
    wget https://launchpad.net/~maarten-fonville/+archive/ubuntu/protobuf/+files/libprotoc-dev_3.6.1-1~maarten0+bionic_amd64.deb -P /tmp && \
    wget https://launchpad.net/~maarten-fonville/+archive/ubuntu/protobuf/+files/libprotobuf-lite17_3.6.1-1~maarten0+bionic_amd64.deb -P /tmp && \
    wget https://launchpad.net/~maarten-fonville/+archive/ubuntu/protobuf/+files/libprotobuf-dev_3.6.1-1~maarten0+bionic_amd64.deb -P /tmp && \
    wget https://launchpad.net/~maarten-fonville/+archive/ubuntu/protobuf/+files/protobuf-compiler_3.6.1-1~maarten0+bionic_amd64.deb -P /tmp && \
    wget https://launchpad.net/~maarten-fonville/+archive/ubuntu/protobuf/+files/googletest_1.9.0.20190831-2~202001251824~ubuntu18.04.1_all.deb -P /tmp && \
    wget https://launchpad.net/~maarten-fonville/+archive/ubuntu/protobuf/+files/libgtest-dev_1.9.0.20190831-2~202001251824~ubuntu18.04.1_amd64.deb -P /tmp && \
    wget https://launchpad.net/~maarten-fonville/+archive/ubuntu/protobuf/+files/google-mock_1.9.0.20190831-2~202001251824~ubuntu18.04.1_amd64.deb -P /tmp &&\
    wget https://launchpad.net/~maarten-fonville/+archive/ubuntu/protobuf/+files/googletest-tools_1.9.0.20190831-2~202001251824~ubuntu18.04.1_amd64.deb -P /tmp &&\
    wget https://launchpad.net/~maarten-fonville/+archive/ubuntu/protobuf/+files/libgmock-dev_1.9.0.20190831-2~202001251824~ubuntu18.04.1_amd64.deb -P /tmp\
    "

    $SUDO sh -c "dpkg -i /tmp/libprotobuf17_3.6.1-1~maarten0+bionic_amd64.deb /tmp/libprotoc17_3.6.1-1~maarten0+bionic_amd64.deb \
    /tmp/libprotoc-dev_3.6.1-1~maarten0+bionic_amd64.deb /tmp/googletest_1.9.0.20190831-2~202001251824~ubuntu18.04.1_all.deb \
    /tmp/libprotobuf-lite17_3.6.1-1~maarten0+bionic_amd64.deb /tmp/libprotobuf-dev_3.6.1-1~maarten0+bionic_amd64.deb \
    /tmp/protobuf-compiler_3.6.1-1~maarten0+bionic_amd64.deb /tmp/libgtest-dev_1.9.0.20190831-2~202001251824~ubuntu18.04.1_amd64.deb \
    /tmp/google-mock_1.9.0.20190831-2~202001251824~ubuntu18.04.1_amd64.deb /tmp/googletest-tools_1.9.0.20190831-2~202001251824~ubuntu18.04.1_amd64.deb \
    /tmp/libgmock-dev_1.9.0.20190831-2~202001251824~ubuntu18.04.1_amd64.deb"

    $SUDO sh -c "rm /tmp/libprotobuf17_3.6.1-1~maarten0+bionic_amd64.deb /tmp/libprotoc17_3.6.1-1~maarten0+bionic_amd64.deb \
    /tmp/libprotoc-dev_3.6.1-1~maarten0+bionic_amd64.deb /tmp/googletest_1.9.0.20190831-2~202001251824~ubuntu18.04.1_all.deb \
    /tmp/libprotobuf-lite17_3.6.1-1~maarten0+bionic_amd64.deb /tmp/libprotobuf-dev_3.6.1-1~maarten0+bionic_amd64.deb \
    /tmp/protobuf-compiler_3.6.1-1~maarten0+bionic_amd64.deb /tmp/libgtest-dev_1.9.0.20190831-2~202001251824~ubuntu18.04.1_amd64.deb \
    /tmp/google-mock_1.9.0.20190831-2~202001251824~ubuntu18.04.1_amd64.deb /tmp/googletest-tools_1.9.0.20190831-2~202001251824~ubuntu18.04.1_amd64.deb \
    /tmp/libgmock-dev_1.9.0.20190831-2~202001251824~ubuntu18.04.1_amd64.deb"

fi

if [ ! -f "/usr/lib/libgtest.a" ]; then 
    $SUDO sh -c "cd /usr/src/googletest/googletest && mkdir -p build && cd build && cmake .. -DCMAKE_CXX_FLAGS=\"-std=c++11\" && make -j"
    $SUDO sh -c "cp /usr/src/googletest/googletest/build/lib/libgtest* /usr/lib/"
    $SUDO sh -c "rm -rf /usr/src/googletest/googletest/build"
    $SUDO sh -c "mkdir /usr/local/lib/googletest"
    $SUDO sh -c "ln -s /usr/lib/libgtest.a /usr/local/lib/googletest/libgtest.a"
    $SUDO sh -c "ln -s /usr/lib/libgtest_main.a /usr/local/lib/googletest/libgtest_main.a"
fi
echo "- libgtest is installed in system."

$SUDO sh -c "apt install git-lfs"
echo "- git-lfs is installed in system."

echo "- Done."