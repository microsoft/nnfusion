FROM nvcr.io/nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

WORKDIR /root

RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential git wget \
  libgtest-dev libprotobuf-dev protobuf-compiler libgflags-dev libsqlite3-dev llvm-dev \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh && \
  bash install_miniconda.sh -b -p /opt/conda && rm install_miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

RUN conda install python~=3.10 pip cmake && conda clean --all

RUN pip install --no-cache-dir --default-timeout=1000 torch==1.12 torchvision==0.13 timm==0.5.4 \
    onnx==1.12 onnxruntime-gpu==1.12 onnxconverter_common==1.12 \
    attrs cloudpickle decorator psutil synr tornado xgboost regex \
    && rm -rf ~/.cache/pip

RUN git clone https://github.com/nox-410/tvm --recursive -b welder \
  && mkdir tvm/build && cd tvm/build && cp ../cmake/config.cmake . \
  && echo "set(USE_LLVM ON)" >> config.cmake && echo "set(USE_CUDA ON)" >> config.cmake \
  && cmake .. && make -j
ENV PYTHONPATH /root/tvm/python:$PYTHONPATH

RUN git clone https://github.com/nox-410/nnfusion -b welder \
  && mkdir nnfusion/build \
  && cd nnfusion/build && cmake .. && make -j
ENV PATH /root/nnfusion/build/src/tools/nnfusion:$PATH

RUN git clone https://github.com/nox-410/cutlass -b welder
ENV CPLUS_INCLUDE_PATH /root/cutlass/include:$CPLUS_INCLUDE_PATH

COPY . welder/
ENV PYTHONPATH /root/welder/python:$PYTHONPATH

CMD bash