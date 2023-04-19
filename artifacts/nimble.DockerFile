FROM nvcr.io/nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

WORKDIR /root

RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh && \
  bash install_miniconda.sh -b -p /opt/conda && rm install_miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

RUN conda install python=3.7 pip numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
RUN conda install -c pytorch magma-cuda110
RUN conda install -c conda-forge onnx
RUN conda install -c conda-forge protobuf=3.9
RUN pip install future

RUN git clone --recursive https://github.com/snuspl/nimble
RUN cd nimble && USE_NINJA=OFF BUILD_TEST=0 USE_DISTRIBUTED=0 USE_NCCL=0 USE_NUMA=0 USE_MPI=0 python setup.py install

RUN pip install --no-dependencies torchvision==0.8 timm pillow

COPY . scripts/

WORKDIR scripts
