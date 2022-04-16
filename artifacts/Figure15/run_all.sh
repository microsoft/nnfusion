SOURCE_DIR=../roller
LOGS_DIR=$SOURCE_DIR/logs
ROLLER_LOGS_DIR=$LOGS_DIR/roller/wo_storage_align/tc_matmul/
CURRENT_DIR=$PWD
TF_DIR=../microbenchmark/tf
TF_LOGS_DIR=$TF_DIR/logs/tc_matmul/

# Step1: reproduce results and generate roller's log file
source ../scripts/profile_tvm_codegen.profile
cd $SOURCE_DIR
bash script_v1/tc_mm.sh

# Step2: reproduce results and generate TF's log file
cd $TF_DIR
bash tc_matmul.sh

# Step3: reproduce ansor and tvm results and collect their logs to autotvm_tensor_core.log
cd $CURRENT_DIR
source ../scripts/profile_tvm.profile
bash run_ansor_autotvm.sh

# Step4: process roller's log files and generate .dat file
cd $CURRENT_DIR
python3.7 process_log_dir.py $ROLLER_LOGS_DIR $TF_LOGS_DIR
python3.7 process_ansor_autotvm_log.py $CURRENT_DIR/logs
python3.7 merge_dat.py

# Step5: plot Figure 15, result: tensorcore_matmul_v100.pdf
gnuplot tensorcore_matmul_v100.plt

