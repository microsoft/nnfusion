# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
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
RUN rm -rf /nnfusion_rt

# Antares Part
RUN apt-get update && apt install -y --no-install-recommends git ca-certificates \
    python3-pip python3-wheel python3-setuptools python3-dev python3-pytest \
    vim less netcat-openbsd inetutils-ping curl patch iproute2 \
    g++ libpci3 libnuma-dev make file openssh-server kmod gdb libopenmpi-dev openmpi-bin \
        autoconf automake autotools-dev libtool llvm-dev screen
RUN [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ] || ln -s /host/usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu
RUN ln -sf libcudart.so /usr/local/cuda/targets/x86_64-linux/lib/libcudart_static.a
RUN git clone https://github.com/microsoft/antares /antares
# Patch
RUN /antares/engine/install_antares_host.sh && rm -rf /var/lib/apt/lists/* ~/.cache
# Run server
RUN echo "" >> /antares/Makefile
RUN echo "host-rest-server:" >> /antares/Makefile
RUN echo "\tbash -c 'trap ctrl_c INT; ctrl_c() { exit 1; }; while true; do BACKEND=\$(BACKEND) HTTP_SERVICE=1 HTTP_PORT=\$(HTTP_PORT) \$(INNER_CMD); done'" >> /antares/Makefile
RUN echo "#!/bin/sh" > /antares/persistant_service.sh
RUN echo "screen -dm bash -c \"cd /antares && BACKEND=c-cuda make host-rest-server\"" >> /antares/persistant_service.sh
RUN echo "screen -dm bash -c \"cd /antares && BACKEND=c-mcpu HTTP_PORT=8881 make host-rest-server\"" >> /antares/persistant_service.sh
RUN echo "exec bash" >> /antares/persistant_service.sh
RUN chmod +x /antares/persistant_service.sh
ENTRYPOINT ["/antares/persistant_service.sh"]