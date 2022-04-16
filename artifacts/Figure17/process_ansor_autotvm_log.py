import sys
import os

runtime_ansor = []
runtime_autotvm = []
op = ["C9", "C12", "C13", "C15", "C28", "C36"]
log_dir = sys.argv[1]

with open(os.path.join(log_dir, "ansor_irregular_conv.log"), "r") as f:
    lines = f.readlines()
    for line in lines:
        if "best runtime: " in line:
            runtime_ansor.append(line.rstrip().split()[-1])

print("irregular_conv runtime ansor: ", len(runtime_ansor))

with open(os.path.join(log_dir, "autotvm_irregular_conv.log"), "r") as f:
    lines = f.readlines()
    for line in lines:
        if "best runtime: " in line:
            runtime_autotvm.append(line.rstrip().split()[-1])
print("irregular_conv runtime autotvm: ", len(runtime_autotvm))

with open("irregular_conv_runtime_ansor_autotvm.dat", "w") as outf:
    outf.write("OP\tAnsor\tAutotvm\n")
    current = 0
    for i in range(6):
        result_str = "\t".join([op[i], str(runtime_ansor[current]), str(runtime_autotvm[current])]) + "\n"
        outf.write(result_str)
        current += 1