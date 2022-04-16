CURRENT_DIR=$PWD

# build docker
bash docker/build_docker.sh

# run into the docker
docker run -it roller/rocm:4.0.1-ubuntu20.04 /bin/bash

# enter this dir
cd /root/nnfusion/artifacts/Table4/src

# build tvm
bash docker/build_tvm.sh

# add TVM path to the environment
echo 'export TVM_HOME=/root/nnfusion/artifacts/.deps/tvm-0.8-codegen\n\
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}\n\' >> /root/.bashrc
source /root/.bashrc

# generate roller logs
bash ./run_roller_op_perf.sh

# add TVM path to the environment
echo 'export TVM_HOME=/root/nnfusion/artifacts/.deps/tvm-0.8\n\
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}\n\' >> /root/.bashrc
source /root/.bashrc

# generate ansor & autotvm logs
bash ./run_ansor_autotvm_op_perf.sh

# generate TF logs
bash ./run_tf_op_perf.sh

