FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV HIP_PLATFORM hcc
ENV PATH $PATH:/opt/rocm/bin
ENV HSA_USERPTR_FOR_PAGED_MEM=0
ENV TF_ROCM_FUSION_ENABLE 1
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/rocm/lib:$LD_LIBRARY_PATH

RUN env > /etc/environment

RUN apt-get update && apt install -y --no-install-recommends git ca-certificates \
    python3-pip python3-wheel python3-setuptools python3-dev python3-pytest \
    vim-tiny less netcat-openbsd inetutils-ping curl patch iproute2 \
    g++ libpci3 libnuma-dev make file openssh-server kmod gdb libopenmpi-dev openmpi-bin psmisc \
        autoconf automake autotools-dev libtool llvm-dev \
        zlib1g-dev rename zip unzip librdmacm-dev gnupg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN printf "deb [arch=amd64] http://repo.radeon.com/rocm/apt/4.0.1/ xenial main" | tee /etc/apt/sources.list.d/rocm_hip.list && \
    apt update --allow-insecure-repositories && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --allow-unauthenticated \
    rocm-dev rocblas hipsparse rccl rocfft rocrand miopen-hip rocthrust hip-rocclr && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN /bin/echo -e "set backspace=indent,eol,start\nset nocompatible\nset ts=4" > /etc/vim/vimrc.tiny

RUN git clone https://github.com/microsoft/nnfusion.git /root/nnfusion --branch osdi22_artifact --single-branch
RUN mkdir /root/nnfusion/artifacts/.deps
RUN git clone https://github.com/apache/tvm /root/nnfusion/artifacts/.deps/tvm-0.8 && cd /root/nnfusion/artifacts/.deps/tvm-0.8 && git checkout 22ba6523cbd14fc44a1b093c482c1d02f3bc4fa5 && git apply /root/nnfusion/artifacts/Table4/src/docker/tvm-autotvm-dense.diff
RUN git clone https://github.com/apache/tvm /root/nnfusion/artifacts/.deps/tvm-0.8-codegen && cd /root/nnfusion/artifacts/.deps/tvm-0.8-codegen && git checkout 22ba6523cbd14fc44a1b093c482c1d02f3bc4fa5 && git apply /root/nnfusion/artifacts/Table4/src/docker/tvm-codegen.diff

RUN pip3 install cloudpickle
RUN pip3 install tensorflow-rocm==1.15.9
RUN apt update
RUN apt install lld-10

