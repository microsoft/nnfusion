# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FROM rocm/dev-ubuntu-18.04:3.8
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

RUN rm -rf /nnfusion_rt
# Antares Part
ENV HIP_PLATFORM hcc
ENV PATH $PATH:/opt/rocm/bin:/usr/local/nvidia/lib64/bin
ENV HSA_USERPTR_FOR_PAGED_MEM=0
ENV TF_ROCM_FUSION_ENABLE 1
ENV HIP_IGNORE_HCC_VERSION=1
RUN env > /etc/environmen

RUN apt-get update && apt install -y --no-install-recommends git ca-certificates \
    python3-pip python3-wheel python3-setuptools python3-dev python3-pytest \
    vim less netcat-openbsd inetutils-ping curl patch iproute2 \
    g++ libpci3 libnuma-dev make file openssh-server kmod gdb libopenmpi-dev openmpi-bin \
        autoconf automake autotools-dev libtool llvm-dev screen

RUN curl -sL http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | apt-key add - && \
    printf "deb [arch=amd64] http://repo.radeon.com/rocm/apt/3.8/ xenial main" | tee /etc/apt/sources.list.d/rocm_hip.list && \
    apt update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    rocm-dev zlib1g-dev unzip librdmacm-dev rocblas hipsparse rccl rocfft rocrand miopen-hip && apt-get clean && rm -rf /var/lib/apt/lists/*
    
RUN git clone https://github.com/microsoft/antares /antares
# Patch
RUN /antares/engine/install_antares_host.sh && rm -rf /var/lib/apt/lists/* ~/.cache
# Run server
RUN echo "" >> /antares/Makefile
RUN echo "host-rest-server:" >> /antares/Makefile
RUN echo "\tbash -c 'trap ctrl_c INT; ctrl_c() { exit 1; }; while true; do BACKEND=\$(BACKEND) HTTP_SERVICE=1 HTTP_PORT=\$(HTTP_PORT) \$(INNER_CMD); done'" >> /antares/Makefile
RUN echo "#!/bin/sh" > /antares/persistant_service.sh
RUN echo "screen -dm bash -c \"cd /antares && BACKEND=c-rocm make host-rest-server\"" >> /antares/persistant_service.sh
RUN echo "screen -dm bash -c \"cd /antares && BACKEND=c-mcpu HTTP_PORT=8881 make host-rest-server\"" >> /antares/persistant_service.sh
RUN echo "exec bash" >> /antares/persistant_service.sh
RUN chmod +x /antares/persistant_service.sh
ENTRYPOINT ["/antares/persistant_service.sh"]