import sys
import os

runtime_ansor = []
runtime_autotvm = []
op = ["128", "256", "512", "1024", "2048", "4096", "8192"]
log_dir = sys.argv[1]

with open(os.path.join(log_dir, "ansor_conv_scale.log"), "r") as f:
    lines = f.readlines()
    for line in lines:
        if "best runtime: " in line:
            runtime_ansor.append(line.rstrip().split()[-1])

print("scale conv runtime ansor: ", len(runtime_ansor))

with open(os.path.join(log_dir, "autotvm_conv_scale.log"), "r") as f:
    lines = f.readlines()
    for line in lines:
        if "best runtime: " in line:
            runtime_autotvm.append(line.rstrip().split()[-1] if not line.rstrip().split()[-1] == "inf" else 0)
print("scale conv runtime autotvm: ", len(runtime_autotvm))

with open("scale_conv_runtime_ansor_autotvm.dat", "w") as outf:
    outf.write("Batch\tAnsor\tAutotvm\n")
    current = 0
    for i in range(7):
        result_str = "\t".join([op[i], str(runtime_ansor[current]), str(runtime_autotvm[current])]) + "\n"
        outf.write(result_str)
        current += 1