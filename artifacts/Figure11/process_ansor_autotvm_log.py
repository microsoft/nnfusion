import sys
import os

compile_time_ansor = []
compile_time_autotvm = []
op = ["C", "D", "E", "M", "P", "R"]
num = [44, 23, 28, 7, 13, 4]
log_dir = sys.argv[1]

with open(os.path.join(log_dir, "ansor_op.log"), "r") as f:
    lines = f.readlines()
    for line in lines:
        if "compilation time: " in line:
            compile_time_ansor.append(line.rstrip().split()[-1])

print("op compilation time ansor:", len(compile_time_ansor))

with open(os.path.join(log_dir, "autotvm_op.log"), "r") as f:
    lines = f.readlines()
    for line in lines:
        if "compilation time: " in line:
            compile_time_autotvm.append(line.rstrip().split()[-1])
print("op compilation time autotvm:", len(compile_time_autotvm))


with open("compile_time_ansor_autotvm.dat", "w") as outf:
    outf.write("OP\tAnsor\tAutotvm\n")
    current = 0
    for i in range(len(op)):
        for j in range(num[i]):
            result_str = "\t".join([op[i] + str(j), str(compile_time_ansor[current]), str(compile_time_autotvm[current])]) + "\n"
            outf.write(result_str)
            current += 1
