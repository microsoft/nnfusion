import sys
import os
import re

roller_log_dir = sys.argv[1]
tf_log_dir = sys.argv[2]
op = sys.argv[3]
benchmarks = ["TF(CudaLib)", "Roller-Top1", "Roller-Top10"]
tf_result_list = {}
roller_result_list = {}
batch_size_list = {}

# process TF log
file_list = os.listdir(tf_log_dir)
for file_name in file_list:
    x_key = op + (file_name[len(op)] \
                if file_name[len(op) + 1] == "_" \
                else file_name[len(op):len(op)+2])
    batch_size = int(re.findall("(_)(\d+)(_)", file_name)[0][1])
    batch_size_list[x_key] = batch_size
    tf_result_list[x_key] = []
    with open(tf_log_dir + file_name) as f:
        for line in f:
            if "ms on avg" in line:
                t = float(line.split()[0])
                tf_result_list[x_key].append(t)

# process roller log
file_list = os.listdir(roller_log_dir)
for file_name in file_list:
    x_key = op + (file_name[len(op)] \
                if file_name[len(op) + 1] == "_" \
                else file_name[len(op):len(op)+2])
    # batch_size = int(re.findall("(_)(\d+)(_)", file_name)[0][1])
    # batch_size_list[x_key] = batch_size
    roller_result_list[x_key] = []
    with open(roller_log_dir + file_name) as f:
        for line in f:
            if "top1 time:" in line:
                t = float(line.rstrip()[11:-2])
                roller_result_list[x_key].append(t)
            if "top10 time:" in line:
                t = float(line.rstrip()[12:-2])
                roller_result_list[x_key].append(t)

with open("scale_test_" + op + ".dat", "w") as outf:
    outf.write("Batch\t{}\n".format("\t".join([str(x) for x in benchmarks])))
    for idx in range(len(batch_size_list)):
        x_key = op + str(idx)
        batch_size = batch_size_list[x_key]
        result_str = "{}\t{}\t{}\n".format(batch_size,
                    "\t".join([str(x) for x in tf_result_list[x_key]]),
                    "\t".join([str(x) for x in roller_result_list[x_key]]),
                    )
        outf.write(result_str)
