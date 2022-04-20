SOURCE_DIR=../roller
LOGS_DIR=$SOURCE_DIR/logs
ROLLER_LOGS_DIR=$LOGS_DIR/roller/wo_storage_align/
CURRENT_DIR=$PWD
TF_DIR=$CURRENT_DIR/../microbenchmark/tf
TF_LOGS_DIR=$TF_DIR/logs/

# Step1: generate Roller kernels for all the 119 operators. 
# Estimated running time: 34min
source ../scripts/profile_tvm_codegen.profile
cd $SOURCE_DIR
time bash script_v1/conv.sh
time bash script_v1/matmul.sh
time bash script_v1/depthwiseconv.sh
time bash script_v1/reduce.sh
time bash script_v1/elementwise.sh
time bash script_v1/pooling.sh

# Step2: reproduce cudalib results and generate TF's log file
# Note: We set the repeated running number of each op as 1,000 here for reproducing in a reasonable time. 
#	However, our paper uses 10,000 to tolerate some variance for TF.
# Estimated running time: 33min
cd $TF_DIR
time bash op_perf.sh

# Step3: reproduce ansor and tvm results and collect their logs to ansor_op.log & autotvm_op.log
# NOTE: Since tuning all kernels with TVM and Ansor will take an extremely long time (e.g., days),
#        this step only processes the tuning traces we run before.
# Estimated running time: 34min

cd $CURRENT_DIR
source ../scripts/profile_tvm.profile
time bash run_ansor_autotvm.sh

# Step4: process log files and collect result data file
cd $CURRENT_DIR
python3.7 process_log_dir.py $ROLLER_LOGS_DIR $TF_LOGS_DIR
python3.7 process_ansor_autotvm_log.py $CURRENT_DIR/logs
python3.7 merge_dat.py
python3.7 Group_data.py

# Step5: plot Figure 10, the figure file name is: op_perf_multi_v100.pdf
gnuplot op_perf_multi_v100.plt
