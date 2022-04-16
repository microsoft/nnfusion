SOURCE_DIR=/home/v-zhuho/TiledCompiler-1/tiled-compiler
LOGS_DIR=$SOURCE_DIR/logs/roller/wo_storage_align/scale/
CURRENT_DIR=$PWD

# Note: this experiment needs to run after Figure12 or Figure 13 experiments

# Step 1: process roller's log files and generate .dat file
cd $CURRENT_DIR
python3.7 -u process_log_dir.py $LOGS_DIR

# Step2: collect ansor and tvm results compilation time logs into 
#     ansor_conv_scale.log
#     ansor_matmul_scale.log
#     autotvm_conv_scale.log
#     autotvm_matmul_scale.log
cd $CURRENT_DIR
source ../scripts/profile_tvm.profile
bash run_ansor_autotvm.sh
python3.7 process_ansor_autotvm_log.py $CURRENT_DIR/logs

# Step3: merge data file and plot figure
python3.7 merge_dat.py
gnuplot scale_compile_time_v100.plt

