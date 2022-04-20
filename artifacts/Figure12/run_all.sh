SOURCE_DIR=../roller
LOGS_DIR=$SOURCE_DIR/logs
ROLLER_LOGS_DIR=$LOGS_DIR/roller/wo_storage_align/scale/matmul/
CURRENT_DIR=$PWD
TF_DIR=$CURRENT_DIR/../microbenchmark/tf
TF_LOGS_DIR=$TF_DIR/logs/scale/matmul/

# Step 1: reproduce results and generate roller's log file
# Estimated running time: 5min
source ../scripts/profile_tvm_codegen.profile
cd $SOURCE_DIR
time bash script_v1/scale_test.sh

# Step 2: reproduce results and generate TF's log file
# Note: We set the repeated running number of each op as 1,000 here for reproducing in a reasonable time. 
#       However, our paper uses 10,000 to tolerate some variance for TF.
# Estimated running time: 8min
cd $TF_DIR
time bash scale_test.sh

# Step 3: reproduce ansor and tvm results and collect their logs to ansor_matmul_scale.log & autotvm_matmul_scale.log
# Estimated running time: 2min
cd $CURRENT_DIR
source ../scripts/profile_tvm.profile
time bash run_ansor_autotvm.sh

# Step 4: process log files and generate result .dat file
cd $CURRENT_DIR
python3.7 process_log_dir.py $ROLLER_LOGS_DIR $TF_LOGS_DIR matmul
python3.7 process_log_dir_ansor_autotvm.py $CURRENT_DIR/logs
python3.7 merge_dat.py

# Step5: plot Figure 12, result: scale_matmul_v100.pdf
gnuplot scale_matmul_v100.plt

