# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FROM rocm/dev-ubuntu-18.04:3.5
RUN rm /etc/apt/sources.list.d/rocm.list
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
RUN wget https://repo.radeon.com/rocm/apt/3.5/pool/main/r/rocblas/rocblas_2.22.0.2367-b2cceba_amd64.deb -P /tmp && dpkg -i /tmp/rocblas_2.22.0.2367-b2cceba_amd64.deb && rm /tmp/rocblas_2.22.0.2367-b2cceba_amd64.deb  
RUN wget https://repo.radeon.com/rocm/apt/3.5/pool/main/m/miopen-hip/miopen-hip_2.4.0.8035-rocm-rel-3.5-30-bd4a330_amd64.deb -P /tmp && dpkg -i /tmp/miopen-hip_2.4.0.8035-rocm-rel-3.5-30-bd4a330_amd64.deb && rm /tmp/miopen-hip_2.4.0.8035-rocm-rel-3.5-30-bd4a330_amd64.deb
RUN apt install -y kmod