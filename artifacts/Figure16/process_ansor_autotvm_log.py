import sys
import os

runtime_ansor = []
runtime_autotvm = []
op = ["M1", "M2"]
log_dir = sys.argv[1]

with open(os.path.join(log_dir, "ansor_small_op.log"), "r") as f:
    lines = f.readlines()
    for line in lines:
        if "best runtime: " in line:
            runtime_ansor.append(line.rstrip().split()[-1])

print("small op runtime ansor: ", len(runtime_ansor))

with open(os.path.join(log_dir, "autotvm_small_op.log"), "r") as f:
    lines = f.readlines()
    for line in lines:
        if "best runtime: " in line:
            runtime_autotvm.append(line.rstrip().split()[-1])
print("small op runtime autotvm: ", len(runtime_autotvm))

with open("small_op_ansor_autotvm.dat", "w") as outf:
    outf.write("OP\tAnsor\tAutotvm\n")
    current = 0
    for i in range(2):
        result_str = "\t".join([op[i], str(runtime_ansor[current]), str(runtime_autotvm[current])]) + "\n"
        outf.write(result_str)
        current += 1