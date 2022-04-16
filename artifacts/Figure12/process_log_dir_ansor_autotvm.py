import sys
import os

runtime_ansor = []
compile_time_ansor = []
runtime_autotvm = []
compile_time_autotvm = []
op = ["512", "1024", "2048", "4096", "8192", "16384"]
log_dir = sys.argv[1]

with open(os.path.join(log_dir, "ansor_matmul_scale.log"), "r") as f:
    lines = f.readlines()
    for line in lines:
        if "best runtime: " in line:
            runtime_ansor.append(line.rstrip().split()[-1])

print("scale matmul runtime ansor:", len(runtime_ansor))

with open(os.path.join(log_dir, "autotvm_matmul_scale.log"), "r") as f:
    lines = f.readlines()
    for line in lines:
        if "best runtime: " in line:
            runtime_autotvm.append(line.rstrip().split()[-1]if not line.rstrip().split()[-1] == "inf" else 0)

print("scale matmul runtime autotvm:", len(runtime_autotvm))

with open("scale_matmul_runtime_ansor_autotvm.dat", "w") as outf:
    outf.write("Batch\tAnsor\tAutotvm\n")
    current = 0
    for i in range(6):
        result_str = "\t".join([op[i], str(runtime_ansor[current]), str(runtime_autotvm[current])]) + "\n"
        outf.write(result_str)
        current += 1
