import sys
import os

runtime_ansor = []
runtime_autotvm = []
op = ["C", "D", "E", "M", "P", "R"]
num = [44, 23, 28, 7, 13, 4]
log_dir = sys.argv[1]

with open(os.path.join(log_dir, "ansor_op.log"), "r") as f:
    lines = f.readlines()
    for line in lines:
        if "best runtime: " in line:
            runtime_ansor.append(line.rstrip().split()[-1])

print("op runtime ansor:", len(runtime_ansor))

with open(os.path.join(log_dir, "autotvm_op.log"), "r") as f:
    lines = f.readlines()
    for line in lines:
        if "best runtime: " in line:
            runtime_autotvm.append(line.rstrip().split()[-1] if not line.rstrip().split()[-1] == "inf" else 0)
print("op runtime autotvm:", len(runtime_autotvm))

with open("runtime_ansor_autotvm.dat", "w") as outf:
    outf.write("OP\tAnsor\tAutotvm\n")
    current = 0
    for i in range(len(op)):
        for j in range(num[i]):
            result_str = "\t".join([op[i] + str(j), str(runtime_ansor[current]), str(runtime_autotvm[current])]) + "\n"
            outf.write(result_str)
            current += 1
