SOURCE_DIR=../roller
LOGS_DIR=$SOURCE_DIR/logs
ROLLER_LOGS_DIR=$LOGS_DIR/roller/wo_storage_align/scale/conv/
CURRENT_DIR=$PWD
TF_DIR=$CURRENT_DIR/../microbenchmark/tf
TF_LOGS_DIR=$TF_DIR/logs/scale/conv/

# Step 1: reproduce results and generate roller's log file
source ../scripts/profile_tvm_codegen.profile
cd $SOURCE_DIR
bash script_v1/scale_test.sh

# Step 2: reproduce results and generate TF's log file
cd $TF_DIR
bash scale_test.sh

# Step 3: reproduce ansor and tvm results and collect their logs to ansor_matmul_scale.log & autotvm_matmul_scale.log
cd $CURRENT_DIR
source ../scripts/profile_tvm.profile
bash run_ansor_autotvm.sh

# Step 4: process log files and generate result .dat file
cd $CURRENT_DIR
python3.7 process_log_dir.py $ROLLER_LOGS_DIR $TF_LOGS_DIR conv
python3.7 process_ansor_autotvm_log.py $CURRENT_DIR/logs
python3.7 merge_dat.py

# Step5: plot Figure 13, result: scale_conv_v100.pdf
gnuplot scale_conv_v100.plt