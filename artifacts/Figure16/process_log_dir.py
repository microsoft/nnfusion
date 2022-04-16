import sys
import os

log_dir = sys.argv[1]
file_list = os.listdir(log_dir)
file_list.sort()

x_key_list = ["M1", "M2"]
result_list = {}
for file_name in file_list:
    x_key = "M" + file_name[6]
    if x_key not in result_list:
        result_list[x_key] = []
    with open(log_dir + file_name) as f:
        for line in f:
            if "top10 time:" in line:
                t = float(line.rstrip()[12:-2])
                result_list[x_key].append(t)

with open("small_op_roller.dat", "w") as outf:
    outf.write("OP\tRoller-O\tRoller-S\n")
    for x_key in x_key_list:
        result_line = result_list[x_key]
        result_str = x_key + "\t" + \
                    "\t".join([str(x) for x in result_line]) + "\n"
        outf.write(result_str)
