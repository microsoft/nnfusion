# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu16.04
# install anaconda && python 3.6

# install CUDA 10.0
RUN apt update && apt install -y cuda-toolkit-10-0 git llvm-6.0 clang-6.0 curl wget
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt install -y git-lfs && git lfs install

RUN git clone https://github.com/microsoft/nnfusion.git /root/nnfusion --branch osdi22_artifact --single-branch

# install anaconda
RUN mkdir /root/nnfusion/artifacts/.deps && curl https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh -o /root/nnfusion/artifacts/.deps/Anaconda3-5.1.0-Linux-x86_64.sh && bash /root/nnfusion/artifacts/.deps/Anaconda3-5.1.0-Linux-x86_64.sh -b -p /root/nnfusion/artifacts/.deps/anaconda3

# install bazel, from https://docs.bazel.build/versions/0.26.0/install-ubuntu.html
RUN apt install -y pkg-config zip g++ zlib1g-dev unzip && curl -L https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-installer-linux-x86_64.sh -o /root/nnfusion/artifacts/.deps/bazel-0.26.1-installer-linux-x86_64.sh && bash /root/nnfusion/artifacts/.deps/bazel-0.26.1-installer-linux-x86_64.sh --prefix=/root/nnfusion/artifacts/.deps/bazel-0.26.1

# Clone source code
RUN git clone https://github.com/apache/tvm /root/nnfusion/artifacts/.deps/tvm-0.8 && cd /root/nnfusion/artifacts/.deps/tvm-0.8 && git checkout 22ba6523cbd14fc44a1b093c482c1d02f3bc4fa5 && git apply /root/nnfusion/artifacts/dockerfile/tvm-autotvm-dense.diff
RUN git clone https://github.com/apache/tvm /root/nnfusion/artifacts/.deps/tvm-0.8-codegen && cd /root/nnfusion/artifacts/.deps/tvm-0.8-codegen && git checkout 22ba6523cbd14fc44a1b093c482c1d02f3bc4fa5 && git apply /root/nnfusion/artifacts/dockerfile/tvm-codegen.diff
RUN git clone https://github.com/tensorflow/tensorflow /root/nnfusion/artifacts/.deps/tensorflow-trt && cd /root/nnfusion/artifacts/.deps/tensorflow-trt && git checkout 5d80e1e8e6ee999be7db39461e0e79c90403a2e4 && cp /root/nnfusion/artifacts/dockerfile/compile_tf_trt7.sh /root/nnfusion/artifacts/.deps/tensorflow-trt/compile.sh && chmod +x /root/nnfusion/artifacts/.deps/tensorflow-trt/compile.sh

# install gnuplot
RUN apt install -y libcairo2-dev libpango1.0-dev
# RUN cd /root/nnfusion/artifacts/.deps/ && wget --no-check-certificate https://versaweb.dl.sourceforge.net/project/gnuplot/gnuplot/5.2.8/gnuplot-5.2.8.tar.gz && tar zxf gnuplot-5.2.8.tar.gz && cd gnuplot-5.2.8 && ./configure && make -j && make install
RUN cd /root/nnfusion/artifacts/.deps/ && wget --no-check-certificate https://versaweb.dl.sourceforge.net/project/gnuplot/gnuplot/5.0.6/gnuplot-5.0.6.tar.gz && tar zxf gnuplot-5.0.6.tar.gz && cd gnuplot-5.0.6 && ./configure && make -j && make install

# install rammer
# To be removed when open-sourced in github
# - Install Requirements 
#RUN bash /root/nnfusion/maint/script/install_dependency.sh
# - Make Install
#RUN cd /root/nnfusion/ && mkdir build && cd build && cmake .. && make -j6 && make install
# - Execute command
#RUN LD_LIBRARY_PATH=/usr/local/lib nnfusion /root/nnfusion/test/models/tensorflow/frozen_op_graph/frozen_abs_graph.pb

# install TensorRT
# from https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
RUN version="7.0.0-1+cuda10.0" && apt install -y libnvinfer7=${version} libnvonnxparsers7=${version} libnvparsers7=${version} libnvinfer-plugin7=${version} libnvinfer-dev=${version} libnvonnxparsers-dev=${version} libnvparsers-dev=${version} libnvinfer-plugin-dev=${version} python-libnvinfer=${version} python3-libnvinfer=${version} && apt-mark hold libnvinfer7 libnvonnxparsers7 libnvparsers7 libnvinfer-plugin7 libnvinfer-dev libnvonnxparsers-dev libnvparsers-dev libnvinfer-plugin-dev python-libnvinfer python3-libnvinfer

RUN echo 'export PATH="$HOME/bin:$HOME/.local/bin:$PATH" \n\
export ARTIFACTS_HOME="/root/nnfusion/artifacts" \n\
export PATH=$ARTIFACTS_HOME/.deps/anaconda3/bin:$PATH \n\
export PATH=$ARTIFACTS_HOME/.deps/bazel-0.26.1/bin:$PATH \n\
export CUDA_HOME=/usr/local/cuda-10.0 \n\
export PATH=$CUDA_HOME/bin${PATH:+:${PATH}} \n\
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} \n\
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/extras/CUPTI/lib64 \n\
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib \n\
export LD_LIBRARY_PATH=.${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} \n\
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ARTIFACTS_HOME/.deps/TensorRT-7.0.0.11/lib \n\
export TENSORRT_HOME=$ARTIFACTS_HOME/.deps/TensorRT-7.0.0.11 \n\
export TVM_HOME=$ARTIFACTS_HOME/.deps/tvm-0.8 \n\
export GNUTERM=dumb \n\
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH} \n\
' >> /root/.bashrc

RUN bash /root/nnfusion/maint/script/install_dependency.sh

RUN ln -s /root/nnfusion/artifacts/.deps/anaconda3/bin/python3 /root/nnfusion/artifacts/.deps/anaconda3/bin/python3.7

RUN mkdir -p /root/nnfusion/artifacts/wheel
RUN cd /root/nnfusion/artifacts/wheel && wget https://github.com/microsoft/nnfusion/raw/osdi20_artifact/artifacts/wheel/tensorflow-1.15.2-cp36-cp36m-linux_x86_64.whl
