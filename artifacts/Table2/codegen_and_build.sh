source .profile

pip install ply

cd frozen_models
bash freeze_models.sh
cd ..

# Roller Kernel Compilation
rm -rf rammer_roller
mkdir rammer_roller

rm -rf $HOME/.cache/nnfusion
mkdir -p $HOME/.cache/nnfusion

mkdir logs

# bert
cd roller_kernels
start_time_bert=$(date +%s)

bash roller_kernelgen_bert_large_infer_bs128.sh

cd ../rammer_roller/
cp ../roller_kernels/bert_kernels/kernel_cache.db $HOME/.cache/nnfusion
nnfusion ../frozen_models/frozen_pbs/frozen_bert_large_infer_bs128.const_folded.pb \
    -f tensorflow -b nnfusion -m graph -fkernel_fusion_level=3 -fblockfusion_level=1 \
    -fconst_folding_backend=CUDA -fwarmup_step=5 -frun_step=100 -fproduct_name="Tesla V100-PCIE-16GB" \
    -fbiasadd_fix=1 -fpattern_substitution=1 -fcodegen_debug=0 -fkernels_as_files=true -fkernels_files_number=60

cd nnfusion_rt/cuda_codegen
cmake .
make -j

end_time_bert=$(date +%s)
cost_time_bert=$[ $end_time_bert-$start_time_bert ]
echo $(($cost_time_bert)) > ../../../logs/compile_time.bert_large_infer_bs128.roller.log

cd ../..
mv nnfusion_rt bert_large_infer_bs128

# nasnet
cd ../roller_kernels
start_time_nas=$(date +%s)

bash roller_kernelgen_nasnet_large_nchw_infer_bs128.sh

cd ../rammer_roller/
cp ../roller_kernels/nasnet_kernels/kernel_cache.db $HOME/.cache/nnfusion
nnfusion ../frozen_models/frozen_pbs/frozen_nasnet_large_nchw_infer_bs128.const_folded.pb \
    -f tensorflow -b nnfusion -m graph -fkernel_fusion_level=3 -fblockfusion_level=1 \
    -fconst_folding_backend=CUDA -fwarmup_step=5 -frun_step=100 -fproduct_name="Tesla V100-PCIE-16GB" \
    -fbiasadd_fix=1 -fcnhw=1 -fpattern_substitution=1 -fcodegen_debug=0 -fkernels_as_files=true -fkernels_files_number=60

cd nnfusion_rt/cuda_codegen
cmake .
make -j

end_time_nas=$(date +%s)
cost_time_nas=$[ $end_time_nas-$start_time_nas ]
echo $(($cost_time_nas)) > ../../../logs/compile_time.nasnet_large_nchw_infer_bs128.roller.log

cd ../..
mv nnfusion_rt nasnet_large_nchw_infer_bs128

# lstm
cd ../roller_kernels
start_time_lstm=$(date +%s)

bash roller_kernelgen_lstm_infer_bs128.sh

cd ../rammer_roller/
cp ../roller_kernels/lstm_kernels/kernel_cache.db $HOME/.cache/nnfusion
nnfusion ../frozen_models/frozen_pbs/frozen_lstm_infer_bs128.const_folded.pb \
    -f tensorflow -b nnfusion -m graph -fkernel_fusion_level=3 -fblockfusion_level=1 \
    -fconst_folding_backend=CUDA -fwarmup_step=5 -frun_step=100 -fproduct_name="Tesla V100-PCIE-16GB" \
    -fbiasadd_fix=1 -fpattern_substitution=1 -fcodegen_debug=0 -fkernels_as_files=true -fkernels_files_number=60

cd nnfusion_rt/cuda_codegen
cmake .
make -j

end_time_lstm=$(date +%s)
cost_time_lstm=$[ $end_time_lstm-$start_time_lstm ]
echo $(($cost_time_lstm)) > ../../../logs/compile_time.lstm_infer_bs128.roller.log

cd ../..
mv nnfusion_rt lstm_infer_bs128

# resnet
cd ../roller_kernels
start_time_res=$(date +%s)
bash roller_kernelgen_resnet50_infer_bs128.sh

cd ../rammer_roller/
cp ../roller_kernels/resnet_kernels/kernel_cache.db $HOME/.cache/nnfusion
nnfusion ../frozen_models/frozen_pbs/frozen_resnet50_infer_bs128.const_folded.pb \
    -f tensorflow -b nnfusion -m graph -fkernel_fusion_level=3 -fblockfusion_level=1 \
    -fconst_folding_backend=CUDA -fwarmup_step=5 -frun_step=100 -fproduct_name="Tesla V100-PCIE-16GB" \
    -fbiasadd_fix=1 -fpattern_substitution=1 -fcodegen_debug=0 -fkernels_as_files=true -fkernels_files_number=60

cd nnfusion_rt/cuda_codegen
cmake .
make -j

end_time_res=$(date +%s)
cost_time_res=$[ $end_time_res-$start_time_res ]
echo $(($cost_time_res)) > ../../../logs/compile_time.resnet50_infer_bs128.roller.log

cd ../..
mv nnfusion_rt resnet50_infer_bs128
