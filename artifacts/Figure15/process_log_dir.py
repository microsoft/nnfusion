import sys
import os

roller_log_dir = sys.argv[1]
tf_log_dir = sys.argv[2]
x_keys = ['M3', 'M4', 'M5', 'M6']
benchmarks = ["TF(CudaLib)", "Roller-Top1", "Roller-Top10"]
tf_result_list = {}
roller_result_list = {}

# process TF log
file_list = os.listdir(tf_log_dir)
file_list.sort(key = lambda x : int(x[5]))
for x_key, file_name in zip(x_keys, file_list):
    tf_result_list[x_key] = []
    with open(tf_log_dir + file_name) as f:
        for line in f:
            if "ms on avg" in line:
                t = float(line.split()[0])
                tf_result_list[x_key].append(t)

# process roller log
file_list = os.listdir(roller_log_dir)
file_list.sort(key = lambda x : int(x[5]))
for x_key, file_name in zip(x_keys, file_list):
    roller_result_list[x_key] = []
    with open(roller_log_dir + file_name) as f:
        for line in f:
            if "top1 time:" in line:
                t = float(line.rstrip()[11:-2])
                roller_result_list[x_key].append(t)
            if "top10 time:" in line:
                t = float(line.rstrip()[12:-2])
                roller_result_list[x_key].append(t)

with open("tensor_core_matmul.dat", "w") as outf:
    outf.write("Op\t{}\n".format("\t".join([str(x) for x in benchmarks])))
    for x_key in x_keys:
        result_str = "{}\t{}\t{}\n".format(x_key,
                    "\t".join([str(x) for x in tf_result_list[x_key]]),
                    "\t".join([str(x) for x in roller_result_list[x_key]]),
                    )
        outf.write(result_str)
