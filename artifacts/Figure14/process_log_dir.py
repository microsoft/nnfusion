import sys
import os

log_dir = sys.argv[1]
ops = ['matmul', 'conv']
x_keys_list = [['M512', 'M1K', 'M2K', 'M4K', 'M8K', 'M16K'],
               ['C128', 'C256', 'C512', 'C1K', 'C2K', 'C4K', 'C8K']]
result_list = []

for op, x_keys in zip(ops, x_keys_list):
    file_list = os.listdir(log_dir + op)
    file_list.sort(key = lambda x : int(x[len(op)]))

    for file_name, x_key in zip(file_list, x_keys):
        result_line = [x_key]
        with open(log_dir + op + '/' + file_name) as f:
            for line in f:
                if "top1 compile time:" in line:
                    t = float(line.rstrip()[18:-2])
                    result_line.append(t)
                if "top10 compile time:" in line:
                    t = float(line.rstrip()[19:-2])
                    result_line.append(t)
        result_list.append(result_line)

with open("scale_compile_time_roller.dat", "w") as outf:
    outf.write("Batch\tRoller-Top1\tRoller-Top10\n")
    for result_line in result_list:
        result_str = "\t".join([str(x) for x in result_line]) + "\n"
        outf.write(result_str)
