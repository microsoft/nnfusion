SOURCE_DIR=../roller
LOGS_DIR=$SOURCE_DIR/logs/roller/wo_storage_align/
CURRENT_DIR=$PWD

# NOTE: This experiment needs to run after the Figure10 experiment!!!

# Step1: process roller's log files and extract compilation time of each op
cd $CURRENT_DIR
python3.7 -u process_log_dir.py $LOGS_DIR


# Step 2: process ansor and tvm log files and extract compilation time of each op
cd $CURRENT_DIR
python3.7 process_ansor_autotvm_log.py $CURRENT_DIR/logs


# Step 3: Merge into single data file
python3.7 merge_dat.py

# Step4: plot Figure 11, file name is: op_compile_time_v100.pdf
gnuplot op_compile_time_v100.plt