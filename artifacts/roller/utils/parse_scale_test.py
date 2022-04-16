import sys
import os
import re

# LOG_DIR=/home/v-zhuho/TiledCompiler-1/tiled-compiler/logs/roller/wo_storage_align/scale/matmul/
log_dir = sys.argv[1]
op = sys.argv[2]
file_list = os.listdir(log_dir)
file_list.sort(key = lambda x : int(x[len(op)]))

result_list = []
for file_name in file_list:
    match = re.findall("(_)(\d+)(_)", file_name)
    batch_size = int(match[0][1])
    result_line = [batch_size]
    with open(log_dir + file_name) as f:
        for line in f:
            if "top1 time:" in line:
                t = float(line.rstrip()[11:-2])
                result_line.append(t)
            if "top10 time:" in line:
                t = float(line.rstrip()[12:-2])
                result_line.append(t)
    result_list.append(result_line)

with open(log_dir + "scale_test_" + op + ".dat", "w") as outf:
    outf.write("Batch\tRoller-Top1\tRoller-Top10\n")
    for result_line in result_list:
        result_str = "\t".join([str(x) for x in result_line]) + "\n"
        outf.write(result_str)