source ../.profile

mkdir -p ~/.cache/nnfusion/
cp ../kernel_db/kernel_cache.db ~/.cache/nnfusion/kernel_cache.db

# rammer_base codegen
rm -rf rammer_base
mkdir rammer_base
cd rammer_base/

nnfusion ../../frozen_models/frozen_pbs/frozen_resnext_nchw_infer_bs1.const_folded.pb -f tensorflow -b nnfusion -m graph -fkernel_fusion_level=3 -fblockfusion_level=0 -fconst_folding_backend=CUDA -fwarmup_step=0 -frun_step=1 -fkernels_as_files=true -fkernels_files_number=60 -fproduct_name="Tesla V100-PCIE-16GB" -fbiasadd_fix=true -fpattern_substitution=true
mv nnfusion_rt resnext_nchw_bs1_rammerbase

nnfusion ../../frozen_models/frozen_pbs/frozen_nasnet_cifar_nchw_infer_bs1.const_folded.pb -f tensorflow -b nnfusion -m graph -fkernel_fusion_level=3 -fblockfusion_level=0 -fconst_folding_backend=CUDA -fwarmup_step=0 -frun_step=1 -fkernels_as_files=true -fkernels_files_number=60 -fproduct_name="Tesla V100-PCIE-16GB" -fbiasadd_fix=true -fpattern_substitution=true
mv nnfusion_rt nasnet_cifar_nchw_bs1_rammerbase

nnfusion ../../frozen_models/frozen_pbs/frozen_alexnet_nchw_infer_bs1.const_folded.pb -f tensorflow -b nnfusion -m graph -fkernel_fusion_level=3 -fblockfusion_level=0 -fconst_folding_backend=CUDA -fwarmup_step=0 -frun_step=1 -fkernels_as_files=true -fkernels_files_number=60 -fproduct_name="Tesla V100-PCIE-16GB" -fbiasadd_fix=true -fpattern_substitution=true
mv nnfusion_rt alexnet_nchw_bs1_rammerbase

nnfusion ../../frozen_models/frozen_pbs/frozen_deepspeech2_infer_bs1.const_folded.pb -f tensorflow -b nnfusion -m graph -fkernel_fusion_level=3 -fblockfusion_level=0 -fconst_folding_backend=CUDA -fwarmup_step=0 -frun_step=1 -fkernels_as_files=true -fkernels_files_number=60 -fdot_transpose=true -fproduct_name="Tesla V100-PCIE-16GB"
mv nnfusion_rt deepspeech2_bs1_rammerbase

nnfusion ../../frozen_models/frozen_pbs/frozen_lstm_infer_bs1.const_folded.pb -f tensorflow -b nnfusion -m graph -fkernel_fusion_level=3 -fblockfusion_level=0 -fconst_folding_backend=CUDA -fwarmup_step=0 -frun_step=1 -fkernels_as_files=true -fkernels_files_number=60 -fdot_transpose=true -fproduct_name="Tesla V100-PCIE-16GB"
mv nnfusion_rt lstm_bs1_rammerbase

nnfusion ../../frozen_models/frozen_pbs/frozen_seq2seq_infer_bs1.const_folded.pb -f tensorflow -b nnfusion -m graph -fkernel_fusion_level=3 -fblockfusion_level=0 -fconst_folding_backend=CUDA -fwarmup_step=0 -frun_step=1 -fkernels_as_files=true -fkernels_files_number=60 -fdot_transpose=true -fproduct_name="Tesla V100-PCIE-16GB"
mv nnfusion_rt seq2seq_bs1_rammerbase

cd ../

# build rammer_base
cd rammer_base/

cd resnext_nchw_bs1_rammerbase/cuda_codegen
cmake .
make -j
cd ../..

cd nasnet_cifar_nchw_bs1_rammerbase/cuda_codegen
cmake .
make -j
cd ../..

cd alexnet_nchw_bs1_rammerbase/cuda_codegen
cmake .
make -j
cd ../..

cd deepspeech2_bs1_rammerbase/cuda_codegen
cmake .
make -j
cd ../..

cd lstm_bs1_rammerbase/cuda_codegen
cmake .
make -j
cd ../..

cd seq2seq_bs1_rammerbase/cuda_codegen
cmake .
make -j
cd ../..

cd ..


# rammer codegen
rm -rf rammer
mkdir rammer
cd rammer/

nnfusion ../../frozen_models/frozen_pbs/frozen_resnext_nchw_infer_bs1.const_folded.pb -f tensorflow -b nnfusion -m graph -fkernel_fusion_level=3 -fblockfusion_level=1 -fconst_folding_backend=CUDA -fwarmup_step=0 -frun_step=1 -fkernels_as_files=true -fkernels_files_number=60 -fproduct_name="Tesla V100-PCIE-16GB" -fbiasadd_fix=true -fpattern_substitution=true
mv nnfusion_rt resnext_nchw_bs1_rammer

nnfusion ../../frozen_models/frozen_pbs/frozen_nasnet_cifar_nchw_infer_bs1.const_folded.pb -f tensorflow -b nnfusion -m graph -fkernel_fusion_level=3 -fblockfusion_level=1 -fconst_folding_backend=CUDA -fwarmup_step=0 -frun_step=1 -fkernels_as_files=true -fkernels_files_number=60 -fproduct_name="Tesla V100-PCIE-16GB" -fbiasadd_fix=true -fpattern_substitution=true
mv nnfusion_rt nasnet_cifar_nchw_bs1_rammer

nnfusion ../../frozen_models/frozen_pbs/frozen_alexnet_nchw_infer_bs1.const_folded.pb -f tensorflow -b nnfusion -m graph -fkernel_fusion_level=3 -fblockfusion_level=1 -fconst_folding_backend=CUDA -fwarmup_step=0 -frun_step=1 -fkernels_as_files=true -fkernels_files_number=60 -fproduct_name="Tesla V100-PCIE-16GB" -fbiasadd_fix=true -fpattern_substitution=true
mv nnfusion_rt alexnet_nchw_bs1_rammer

nnfusion ../../frozen_models/frozen_pbs/frozen_deepspeech2_infer_bs1.const_folded.pb -f tensorflow -b nnfusion -m graph -fkernel_fusion_level=3 -fblockfusion_level=1 -fconst_folding_backend=CUDA -fwarmup_step=0 -frun_step=1 -fkernels_as_files=true -fkernels_files_number=60 -fdot_transpose=true -fproduct_name="Tesla V100-PCIE-16GB"
mv nnfusion_rt deepspeech2_bs1_rammer

nnfusion ../../frozen_models/frozen_pbs/frozen_lstm_infer_bs1.const_folded.pb -f tensorflow -b nnfusion -m graph -fkernel_fusion_level=3 -fblockfusion_level=1 -fconst_folding_backend=CUDA -fwarmup_step=0 -frun_step=1 -fkernels_as_files=true -fkernels_files_number=60 -fdot_transpose=true -fproduct_name="Tesla V100-PCIE-16GB"
mv nnfusion_rt lstm_bs1_rammer

nnfusion ../../frozen_models/frozen_pbs/frozen_seq2seq_infer_bs1.const_folded.pb -f tensorflow -b nnfusion -m graph -fkernel_fusion_level=3 -fblockfusion_level=1 -fconst_folding_backend=CUDA -fwarmup_step=0 -frun_step=1 -fkernels_as_files=true -fkernels_files_number=60 -fdot_transpose=true -fproduct_name="Tesla V100-PCIE-16GB"
mv nnfusion_rt seq2seq_bs1_rammer

cd ../

# build rammer
cd rammer/

cd resnext_nchw_bs1_rammer/cuda_codegen
cmake .
make -j
cd ../..

cd nasnet_cifar_nchw_bs1_rammer/cuda_codegen
cmake .
make -j
cd ../..

cd alexnet_nchw_bs1_rammer/cuda_codegen
cmake .
make -j
cd ../..

cd deepspeech2_bs1_rammer/cuda_codegen
cmake .
make -j
cd ../..

cd lstm_bs1_rammer/cuda_codegen
cmake .
make -j
cd ../..

cd seq2seq_bs1_rammer/cuda_codegen
cmake .
make -j
cd ../..

cd ../

