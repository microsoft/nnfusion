# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FROM rocm/dev-ubuntu-18.04:3.5
RUN apt update && apt install -y git
RUN git clone https://github.com/microsoft/nnfusion.git /root/nnfusion --branch master --single-branch
# - Install Requirements 
RUN bash /root/nnfusion/maint/script/install_dependency.sh
# - Make Install
RUN cd /root/nnfusion/ && mkdir build && cd build && cmake .. && make -j6 && make install
# - Execute command
RUN LD_LIBRARY_PATH=/usr/local/lib nnfusion /root/nnfusion/test/models/tensorflow/frozen_op_graph/frozen_abs_graph.pb
RUN apt install -y python3 python3-pip
RUN pip3 install numpy
RUN chmod +x /usr/local/bin/templates/rocm_adapter/hipify-nnfusion
RUN ln -s /opt/rocm/bin/hipcc /opt/rocm/bin/hcc
RUN apt install -y rocblas miopen-hip kmod
# sudo docker run -it --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video nnfusion/rocm/dev-ubuntu-18.04:3.5