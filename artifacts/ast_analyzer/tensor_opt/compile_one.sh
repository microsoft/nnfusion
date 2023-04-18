#!/bin/bash
set -o pipefail
set -e
rm -r $1 || true
mkdir -p $1
cd $1
if [ ! -d "bin" ]; then
    ln -s ../bin bin
fi
nnfusion ../forward.onnx ${@:2} &> codegen.log
cd nnfusion_rt/^^RT_DIR
cmake . &> ../../cmake.log
make &> ../../make.log
