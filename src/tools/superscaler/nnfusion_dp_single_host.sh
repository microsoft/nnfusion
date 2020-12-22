# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License

#!/bin/bash
# saner programming env: these switches turn some bugs into errors
set -o errexit -o pipefail -o noclobber -o nounset

#TODO: test whether superscaler is installed
#TODO: add cmdline arg parser
#example
#./nnfusion_dp_single_host_train.sh  data/mnist_mlp.onnx "-f onnx -p \"batch:3\" -fautodiff -ftraining_mode -fextern_result_memory=True" 10.0.0.25:2  data/resource_pool.yaml

#user specified inputs
MODEL_FILE_PATH=$(realpath $1)
MODEL_COMPILING_OPTIONS="$2"
#here, we only support one host mutilple worker
DEPLOYMENT_CONFIG=$3
CLUSTER_SPEC_FILE=$(realpath $4)

OLD_PWD=$(pwd)
echo $PWD
NUM_PROCS_SAME_HOST=${DEPLOYMENT_CONFIG##*:}
NNFUSION_EXE='./src/tools/nnfusion/nnfusion'
DUMPED_GRAPH_FILENAME='graph'
SUPERSCALER_COMPILING_OPTIONS="-fadd_sc_allreduce=true -fenable_export_graph=true -fnnfusion_graph_path=$DUMPED_GRAPH_FILENAME.pb"
COMPILE_CMD="$NNFUSION_EXE $MODEL_FILE_PATH $MODEL_COMPILING_OPTIONS $SUPERSCALER_COMPILING_OPTIONS"

echo "-- Creating build dir for compiling NNFusion model"
rm -fr build && mkdir build && cd build && cp -r ../nnf_py . 
pushd ../../../../../.. > /dev/null
echo "-- Building NNFusion"
rm -fr build && mkdir build && cd build && cmake .. > /dev/null
make -j > /dev/null || make
echo "-- Compiling model $MODEL_FILE_PATH"
eval $COMPILE_CMD > /dev/null
echo "-- Dumping NNFusion graph"
python3 src/tools/serialize/nnfusion_serialize_tool.py $DUMPED_GRAPH_FILENAME.pb $DUMPED_GRAPH_FILENAME.pbtxt > /dev/null 2>&1
#copy generated resources
cp -r nnfusion_rt/cuda_codegen/* $OLD_PWD/build
cp -r $DUMPED_GRAPH_FILENAME.pbtxt $OLD_PWD/build
popd > /dev/null

#modify cmakelist
sed  -i 's/.*cuda_add_library.*/set(CMAKE_POSITION_INDEPENDENT_CODE ON)\ncuda_add_library(${TARGET_NAME} SHARED ${SRC})/' CMakeLists.txt

echo "-- Generating superscaler runnig plan"
python3 -c "from superscaler.nnfusion import generate_data_parallelism_plan; \
           generate_data_parallelism_plan(\"$DUMPED_GRAPH_FILENAME.pbtxt\", \
                                   ${NUM_PROCS_SAME_HOST}, \
                                   \"${CLUSTER_SPEC_FILE}\", \
                                   \"${PWD}\", \
                                   communication_DSL='ring')" > /dev/null 


MPIRUN=$(which mpirun)
LAUNCH_CMD="$MPIRUN \
--tag-output \
--output-filename .result\
"
for i in $(seq 0 $(($NUM_PROCS_SAME_HOST-1)))
do 
    if [ "$i" == "0" ];then
                LAUNCH_CMD="$LAUNCH_CMD -np 1 -host $DEPLOYMENT_CONFIG python nnf_py/train.py "$i/plan.json" "       
        else
                LAUNCH_CMD="$LAUNCH_CMD  : -np 1 -host $DEPLOYMENT_CONFIG python nnf_py/train.py "$i/plan.json" "
    fi
done

echo "-- Writing launch command into train.sh "
echo $LAUNCH_CMD > train.sh && chmod +x train.sh
echo "-- Compile successfully"



