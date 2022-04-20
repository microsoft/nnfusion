SOURCE_DIR=../roller
LOGS_DIR=$SOURCE_DIR/logs/roller/wo_storage_align/conv_breakdown/
CURRENT_DIR=$PWD

# Step1: reproduce results and generate roller's log file
# Estimated running time: 40min
source ../scripts/profile_tvm_codegen.profile
cd $SOURCE_DIR
time bash script_v1/conv_breakdown.sh

# Step2: reproduce ansor and tvm results and collect their logs to ansor_irregular_conv.log & autotvm_irregular_conv.log
# Estimated running time: 2min
cd $CURRENT_DIR
source ../scripts/profile_tvm.profile
time bash run_ansor_autotvm.sh

# Step3: process roller's log files and generate .dat file
cd $CURRENT_DIR
python3.7 process_log_dir.py $LOGS_DIR
python3.7 process_ansor_autotvm_log.py $CURRENT_DIR/logs
python3.7 merge_dat.py

# Step4: plot Figure 17, result: irregular_shape_v100.pdf
gnuplot irregular_shape_v100.plt

