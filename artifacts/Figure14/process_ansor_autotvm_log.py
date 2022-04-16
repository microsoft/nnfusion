import sys
import os

compile_time_ansor = []
compile_time_autotvm = []
op = ["M512", "M1K", "M2K", "M4K", "M8K", "M16K", "C128", "C256", "C512", "C1K", "C2K", "C4K", "C8K"]
log_dir = sys.argv[1]


with open(os.path.join(log_dir, "ansor_matmul_scale.log"), "r") as f:
    lines = f.readlines()
    for line in lines:
        if "compilation time: " in line:
            compile_time_ansor.append(line.rstrip().split()[-1])

with open(os.path.join(log_dir, "autotvm_matmul_scale.log"), "r") as f:
    lines = f.readlines()
    for line in lines:
        if "compilation time: " in line:
            compile_time_autotvm.append(line.rstrip().split()[-1])

with open(os.path.join(log_dir, "ansor_conv_scale.log"), "r") as f:
    lines = f.readlines()
    for line in lines:
        if "compilation time: " in line:
            compile_time_ansor.append(line.rstrip().split()[-1])


with open(os.path.join(log_dir, "autotvm_conv_scale.log"), "r") as f:
    lines = f.readlines()
    for line in lines:
        if "compilation time: " in line:
            compile_time_autotvm.append(line.rstrip().split()[-1])

print("scale compilation ansor:", len(compile_time_ansor))
print("scale compilation auotvm:", len(compile_time_autotvm))


with open("scale_compile_time_ansor_autotvm.dat", "w") as outf:
    outf.write("Batch\tAnsor\tAutotvm\n")
    current = 0
    for i in range(13):
        result_str = "\t".join([op[i], str(compile_time_ansor[current]), str(compile_time_autotvm[current])]) + "\n"
        outf.write(result_str)
        current += 1