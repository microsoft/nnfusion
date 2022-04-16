import sys
import os

roller_log_dir = sys.argv[1]
tf_log_dir = sys.argv[2]
ops = {'matmul': "M", 'conv': "C", 'depthwiseconv': "D", 'elementwise': "E", 'pooling': "P", 'reduce': "R"}
benchmarks = ["TF(CudaLib)", "Roller-Top1", "Roller-Top10"]
tf_result_list = {}
roller_result_list = {}
op_num = {}

# process TF log
for op in ops:
    file_list = os.listdir(tf_log_dir + op)
    op_num[op] = len(file_list)
    for file_name in file_list:
        x_key = op + (file_name[len(op)] \
                    if file_name[len(op) + 1] == "_" \
                    else file_name[len(op):len(op)+2])
        tf_result_list[x_key] = []
        with open(tf_log_dir + op + '/' + file_name) as f:
            for line in f:
                if "ms on avg" in line:
                    t = float(line.split()[0])
                    tf_result_list[x_key].append(t)

# process roller log
for op in ops:
    file_list = os.listdir(roller_log_dir + op)
    for file_name in file_list:
        x_key = op + (file_name[len(op)] \
                    if file_name[len(op) + 1] == "_" \
                    else file_name[len(op):len(op)+2])
        roller_result_list[x_key] = []
        with open(roller_log_dir + op + '/' + file_name) as f:
            for line in f:
                if "top1 time:" in line:
                    t = float(line.rstrip()[10:-2])
                    roller_result_list[x_key].append(t)
                if "top10 time:" in line:
                    t = float(line.rstrip()[11:-2])
                    roller_result_list[x_key].append(t)

with open("kernel_time_all.dat", "w") as outf:
    outf.write("OP\t{}\n".format("\t".join([str(x) for x in benchmarks])))
    for op in ops:
        for idx in range(op_num[op]):
            x_key = op + str(idx)
            result_str = "{}\t{}\t{}\n".format(ops[op] + str(idx),
                        "\t".join([str(x) for x in tf_result_list[x_key]]),
                        "\t".join([str(x) for x in roller_result_list[x_key]]),
                        )
            outf.write(result_str)
