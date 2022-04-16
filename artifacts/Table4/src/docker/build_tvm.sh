echo "Build tvm-0.8"
tvmfolder=/root/nnfusion/artifacts/.deps/tvm-0.8
cd $tvmfolder && git submodule init && git submodule update && mkdir build && cd build && cp /root/nnfusion/artifacts/Table4/src/docker/tvm-config.cmake ./config.cmake && cmake .. && make -j
apt-get -y install antlr4
pip install tornado==4.5.3 psutil==5.4.3 xgboost==0.90 decorator==4.2.1 attrs==17.4.0 mypy==0.720 orderedset==2.0.1 antlr4-python3-runtime==4.7.2
pip install tflearn
pip install scipy==1.1.0

echo "Build tvm-0.8-codegen"
tvmfoldercg=/root/nnfusion/artifacts/.deps/tvm-0.8-codegen
cd $tvmfoldercg && git submodule init && git submodule update && mkdir build && cd build && cp /root/nnfusion/artifacts/Table4/src/docker/tvm-config-codegen.cmake ./config.cmake && cmake .. && make -j