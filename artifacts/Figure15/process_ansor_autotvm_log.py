import sys
import os

runtime_ansor = []
runtime_autotvm = []
op = ["M3", "M4", "M5", "M6"]
log_dir = sys.argv[1]

with open(os.path.join(log_dir, "autotvm_tensor_core.log"), "r") as f:
    lines = f.readlines()
    for line in lines:
        if "best runtime: " in line:
            runtime_autotvm.append(line.rstrip().split()[-1] if not line.rstrip().split()[-1] == "inf" else 0)
print("op runtime autotvm:", len(runtime_autotvm))

with open("tensor_core_runtime_autotvm.dat", "w") as outf:
    outf.write("Batch\tAutotvm\n")
    current = 0
    for i in range(4):
        result_str = "\t".join([op[i], str(runtime_autotvm[current])]) + "\n"
        outf.write(result_str)
        current += 1
