import sys
import os

roller_log_dir = sys.argv[1]
tf_log_dir = sys.argv[2]
autotvm_log_name = sys.argv[3]
ansor_log_name = sys.argv[4]

tf_result_dict = {}
tf_result_list = []
roller_result_list = {}
op_num = {}
autotvm_result_list = []
ansor_result_list = []

ops = {
    'conv': "conv", 
    'depthwiseconv': "depthwiseconv", 
    'elementwise': "relu", 
    'pooling': "pooling", 
    'reduce': "reduce",
    'matmul': "matmul", 
}
benchmarks = [
    "TF(CudaLib)", 
    "TVM", 
    "Ansor"
]
comp_list = [
    "Better Performance",
    "Perf. within 5%",
    "Perf. within 10%",
    "Perf. within 50%",
    "Perf. within 90%",
]

# process roller log
for op in ops:
    file_list = os.listdir(roller_log_dir + op)
    # print(op, file_list)
    op_num[op] = len(file_list)
    for file_name in file_list:
        x_key = op + (file_name[len(ops[op])] \
                    if file_name[len(ops[op]) + 1] == "_" \
                    else file_name[len(ops[op]):len(ops[op])+2])
        with open(roller_log_dir + op + '/' + file_name) as f:
            for line in f:
                if "top10 time:" in line:
                    t = float(line.rstrip()[11:-2])
                    roller_result_list[x_key] = t

# process TF log
for op in ops:
    file_list = os.listdir(tf_log_dir + op)
    prefix = file_list[0].split('_')[0]
    file_num = len(file_list)
    for i in range(file_num):
        file_name = tf_log_dir + op + '/{}_{}.log'.format(prefix, i)
        with open(file_name, 'r') as f:
            for line in f:
                if "ms on avg" in line:
                    t = float(line.split()[0])
                    tf_result_list.append(t)
        # print(file_name)

# process ansor log
with open(ansor_log_name) as f:
    for line in f:
        if "best runtime:" in line:
            ansor_result_list.append(float(line.rstrip().split()[-1]))

# process autotvm log
with open(autotvm_log_name) as f:
    for line in f:
        if "best runtime:" in line:
            autotvm_result_list.append(float(line.rstrip().split()[-1]))

# generate result
print("-------------------------Table3-------------------------\n")
print("{:20s}{:12s}{:12s}{:12s}".format("", 
            benchmarks[0], benchmarks[1], benchmarks[2]))
better_total = {}
for comp in comp_list:
    result_list = []
    for benchmark in benchmarks:
        is_true = 0
        total = 0
        op_idx = 0
        for op in ops:
            for idx in range(op_num[op]):
                total += 1
                x_key = op + str(idx)
                if benchmark == "TF(CudaLib)":
                    base_perf = tf_result_list[op_idx]
                if benchmark == "TVM":
                    base_perf = autotvm_result_list[op_idx]
                if benchmark == "Ansor":
                    base_perf = ansor_result_list[op_idx]
                roller_perf = roller_result_list[x_key]
                
                if comp == "Better Performance":
                    if roller_perf < base_perf:
                        is_true += 1
                if comp == "Perf. within 5%":
                    if (roller_perf >= base_perf) and (((roller_perf - base_perf) / roller_perf) < 0.05):
                        is_true += 1
                if comp == "Perf. within 10%":
                    if (roller_perf >= base_perf) and (((roller_perf - base_perf) / roller_perf) < 0.1):
                        is_true += 1
                if comp == "Perf. within 50%":
                    if (roller_perf >= base_perf) and (((roller_perf - base_perf) / roller_perf) < 0.5):
                        is_true += 1
                if comp == "Perf. within 90%":
                    if (roller_perf >= base_perf) and (((roller_perf - base_perf) / roller_perf) < 0.9):
                        is_true += 1
                op_idx += 1
        if comp == "Better Performance":
            better_total[benchmark] = is_true
        else:
            is_true += better_total[benchmark]
        
        result_list.append((is_true / total) * 100)

    result_str = "{:20s}{:9.1f}%{:9.1f}%{:9.1f}%\n".format(comp, 
                        result_list[0], result_list[1], result_list[2])
    print(result_str)

