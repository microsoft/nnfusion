# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FROM ubuntu:20.04
RUN apt update && apt install -y git
RUN git clone https://github.com/microsoft/nnfusion.git /root/nnfusion --branch master --single-branch
# - Install Requirements (set noninteractive to skip time zone setting for tzdata)
RUN DEBIAN_FRONTEND="noninteractive" bash /root/nnfusion/maint/script/install_dependency.sh
# - Make Install
RUN cd /root/nnfusion/ && mkdir build && cd build && cmake .. && make -j6 && make install
# - Execute command
RUN LD_LIBRARY_PATH=/usr/local/lib nnfusion /root/nnfusion/test/models/tensorflow/frozen_op_graph/frozen_abs_graph.pb
