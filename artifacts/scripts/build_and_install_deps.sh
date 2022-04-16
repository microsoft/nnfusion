# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# build and install tensorflow-1.15.2 with TensorRT-7.0 support

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


tftrt=$THIS_SCRIPT_DIR/../wheel/tensorflow-1.15.2-cp36-cp36m-linux_x86_64.whl

# build and install tf-trt
if [ ! -f "$tftrt" ]; then
    echo "Build Tensorflow 1.15.2 with TensorRT 7.0.0 ..."
    cd $THIS_SCRIPT_DIR/../.deps/tensorflow-trt
    #git clone https://github.com/tensorflow/tensorflow.git tensorflow-trt
    #cd tensorflow-trt
    #git checkout v1.15.2
    pip install keras_preprocessing
    ./configure # see configure.screensnap
    bash compile.sh
    cp tensorflow-1.15.2-cp36-cp36m-linux_x86_64.whl $tftrt
fi

pip install $tftrt

# build and install tvm
echo "Build tvm-0.8"
tvmfolder=$THIS_SCRIPT_DIR/../.deps/tvm-0.8
cd $tvmfolder && git submodule init && git submodule update && mkdir build && cd build && cp $THIS_SCRIPT_DIR/tvm-config.cmake ./config.cmake && cmake .. && make -j
apt-get -y install antlr4
pip install tornado==4.5.3 psutil==5.4.3 xgboost==0.90 decorator==4.2.1 attrs==17.4.0 mypy==0.720 orderedset==2.0.1 antlr4-python3-runtime==4.7.2
pip install tflearn
pip install scipy==1.1.0

echo "Build tvm-0.8-codegen"
tvmfoldercg=$THIS_SCRIPT_DIR/../.deps/tvm-0.8-codegen
cd $tvmfoldercg && git submodule init && git submodule update && mkdir build && cd build && cp $THIS_SCRIPT_DIR/tvm-config.cmake ./config.cmake && cmake .. && make -j

# build and install rammer
echo "Build Rammer(NNFusion)"
# bash $THIS_SCRIPT_DIR/../../maint/script/install_dependency.sh
bash $THIS_SCRIPT_DIR/../../maint/script/build.sh
