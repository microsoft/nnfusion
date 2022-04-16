import sys
import os

log_dir = sys.argv[1]
ops = ['matmul', 'conv', 'depthwiseconv', 'elementwise', 'pooling', 'reduce']
top1_result_list = []
top10_result_list = []

for op in ops:
    file_list = os.listdir(log_dir + op)

    for file_name in file_list:
        with open(log_dir + op + '/' + file_name) as f:
            for line in f:
                if "top1 compile time:" in line:
                    t = float(line.rstrip()[18:-2])
                    top1_result_list.append(t)
                if "top10 compile time:" in line:
                    t = float(line.rstrip()[19:-2])
                    top10_result_list.append(t)

top1_result_list.sort()
top10_result_list.sort()

with open("compile_time_roller.dat", "w") as outf:
    outf.write("OP\tRoller-Top1\tRoller-Top10\n")
    idx = 0
    for t1, t10 in zip(top1_result_list, top10_result_list):
        result_str = "\t".join([str(idx), str(t1), str(t10)]) + "\n"
        outf.write(result_str)
        idx += 1
