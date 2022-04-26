current_dir=$PWD
ansor_cc=$current_dir/../microbenchmark/tvm/e2e/ansor
ansor_log=$current_dir/../microbenchmark/tvm/e2e/ansor_log
autotvm_cc=$current_dir/../microbenchmark/tvm/e2e/autotvm
autotvm_log=$current_dir/../microbenchmark/tvm/e2e/autotvm_log
ansor_source_code=$current_dir/../microbenchmark/tvm/ansor
script_dir=$current_dir/../microbenchmark/tvm/script/e2e
autotvm_source_code=$current_dir/../microbenchmark/tvm/autotvm
output_log_dir=$current_dir/logs
exe_dir=$current_dir/ansor_autotvm_exe

nnfusion=~/nnfusion/build/src/tools/nnfusion/nnfusion
model_dir=$current_dir/frozen_models/frozen_pbs
flag="-f tensorflow -b nnfusion -m graph -fkernel_fusion_level=3 -fblockfusion_level=1 -fconst_folding_backend=CUDA -fwarmup_step=5 -frun_step=100 -fkernels_as_files=true -fkernels_files_number=60 -fproduct_name=\"Tesla V100-PCIE-16GB\" -fpattern_substitution=1 -fbiasadd_fix=0"
bert_model=${model_dir}/frozen_bert_large_infer_bs128.const_folded.pb
lstm_model=${model_dir}/frozen_lstm_infer_bs128.const_folded.pb
nasnet_model=${model_dir}/frozen_nasnet_large_nchw_infer_bs128.const_folded.pb
resnet_model=${model_dir}/frozen_resnet50_infer_bs128.const_folded.pb


function benchmark() {
    model=$1
    type=$2
    type_log=$2_log
    type_cc=$2_cc
    type_source_code=$2_source_code
    model_url=$1_model

    # generate cc
    bash ${script_dir}/bench0_${model}.sh ${!type_source_code} ${!type_log} > ${output_log_dir}/compile_time_${model}_${type}.log
    cp ${!type_log}/*.cc ${!type_cc}
    

    # kernel injection
    rm ~/.cache/* -rf
    bash ${script_dir}/${model}_injection.sh ${!type_cc} ${type}

    # codegen the model
    $nnfusion ${!model_url} $flag >> ${output_log_dir}/run_nnfusion_${model}_${type}.log


    mv nnfusion_rt $exe_dir/nnfusion_${model}_${type}
    cd $exe_dir/nnfusion_${model}_${type}/cuda_codegen

    start_time=$(date +%s)

    cmake .
    make -j

    end_time=$(date +%s)
    cost_time=$[ $end_time-$start_time ]
    echo "cost time:" $cost_time >> ${output_log_dir}/run_nnfusion_${model}_${type}.log
    CUDA_VISIBLE_DEVICES=0 ./main_test >> ${output_log_dir}/run_nnfusion_${model}_${type}.log
    cd $current_dir
}


mkdir -p $output_log_dir
mkdir -p $exe_dir

benchmark bert ansor
benchmark bert autotvm
benchmark lstm ansor
benchmark lstm autotvm
benchmark resnet ansor
benchmark resnet autotvm
benchmark nasnet ansor
benchmark nasnet autotvm
