ansor_cc=/home/shanbinke/TiledCompiler/tiled-compiler/microbenchmark/tvm/e2e/ansor_test
ansor_log=/home/shanbinke/TiledCompiler/tiled-compiler/microbenchmark/tvm/e2e/ansor_log
autotvm_cc=/home/shanbinke/TiledCompiler/tiled-compiler/microbenchmark/tvm/e2e/autotvm_test
autotvm_log=/home/shanbinke/TiledCompiler/tiled-compiler/microbenchmark/tvm/e2e/autotvm_log
ansor_source_code=/home/shanbinke/TiledCompiler/tiled-compiler/microbenchmark/tvm/ansor
autotvm_source_code=/home/shanbinke/TiledCompiler/tiled-compiler/microbenchmark/tvm/autotvm

nnfusion=~/nnfusion/build/src/tools/nnfusion/nnfusion
model_dir=/home/lingm/projects0/sokoban/frozen_models_for_op_bench
flag="-f tensorflow -b nnfusion -m graph -fkernel_fusion_level=3 -fblockfusion_level=1 -fconst_folding_backend=CUDA -fwarmup_step=5 -frun_step=1000 -fkernels_as_files=true -fkernels_files_number=60 -fproduct_name=\"Tesla V100-PCIE-16GB\" -fpattern_substitution=1"
bert_model=${model_dir}/frozen_lstm_infer_bs128.const_folded.pb
lstm_model=${model_dir}/frozen_lstm_infer_bs128.const_folded.pb
nasnet_model=${model_dir}/frozen_nasnet_large_nchw_infer_bs128.const_folded.pb
resnet_model=${model_dir}/frozen_resnet50_infer_bs128.const_folded.pb


function benchmark() {
    current=$(pwd)
    model=$1
    type=$2
    type_log=$2_log
    type_cc=$2_cc
    type_source_code=$2_source_code
    model_url=$1_model

    # generate cc
    # bash bench0_${model}.sh ${!type_source_code} ${!type_log} > runtime_compile_${model}_${type}.txt
    # cp ${!type_log}/*.cc ${!type_cc}
    

    # kernel injection
    rm ~/.cache/* -rf
    bash ${model}_injection.sh ${!type_cc} ${type}

    # codegen the model
    $nnfusion ${!model_url} $flag >> run_nnfusion_${model}_${type}

    # mv nnfusion_rt nnfusion_${model}_${type}
    cd nnfusion_${model}_${type}/cuda_codegen

    start_time=$(date +%s)

    cmake .
    make -j

    end_time=$(date +%s)
    cost_time=$[ $end_time-$start_time ]
    echo "cost time:" $cost_time >> run_nnfusion_${model}_${type}
    ./main_test >> ${current}/run_nnfusion_${model}_${type}
    cd $current
}

# benchmark bert ansor
# benchmark bert autotvm
# benchmark lstm ansor
# benchmark lstm autotvm
# benchmark resnet ansor
# benchmark resnet autotvm
benchmark nasnet ansor
# benchmark nasnet autotvm
