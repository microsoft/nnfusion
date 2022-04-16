

print("----------------------------Table2----------------------------\n")
print("{:20s}{:12s}{:12s}{:12s}{:12s}".format("", 
            'BERT-Large', 'ResNet', 'NASNet', 'LSTM'))

def get_tf_avg_time(line):
    return float(line.split(']')[-2].split(', ')[-1])

model_name_list = ['bert_large_infer_bs128', 'resnet50_infer_bs128', 'nasnet_large_nchw_infer_bs128', 'lstm_infer_bs128']
model_name_list_ansor_autotvm = ['bert', 'resnet', 'nasnet', 'lstm']

tf_time = []
for model in model_name_list:
    with open('logs/{}.tf.1000.log'.format(model), 'r') as f:
        for line in f:
            if 'Summary: [min, max, mean]' in line:
                tf_time.append(get_tf_avg_time(line))
print("{:20s}{:12s}{:12s}{:12s}{:12s}".format("TF", 
            str(tf_time[0]), str(tf_time[1]), str(tf_time[2]), str(tf_time[3])))

xla_time = []
for model in model_name_list:
    # TODO OOM
    with open('logs/{}.xla.1000.log'.format(model), 'r') as f:
        for line in f:
            if 'Summary: [min, max, mean]' in line:
                xla_time.append(get_tf_avg_time(line))
print("{:20s}{:12s}{:12s}{:12s}{:12s}".format("TF-XLA", 
            str(xla_time[0]), str(xla_time[1]), str(xla_time[2]), str(xla_time[3])))

trt_time = []
for model in model_name_list:
    if 'bert' in model: 
        trt_time.append('N/A')
        continue
    with open('logs/{}.trt.1000.log'.format(model), 'r') as f:
        for line in f:
            if 'Summary: [min, max, mean]' in line:
                trt_time.append(get_tf_avg_time(line))
print("{:20s}{:12s}{:12s}{:12s}{:12s}".format("TF-TRT", 
            str(trt_time[0]), str(trt_time[1]), str(trt_time[2]), str(trt_time[3])))

# TODO ansor time

# TODO rammer + tvm time
rammer_autotvm_time = []
for model in model_name_list_ansor_autotvm:
    with open('logs/run_nnfusion_{}_autotvm.log'.format(model), 'r') as f:
        for line in f:
            if 'Summary: [min, max, mean]' in line:
                rammer_autotvm_time.append(get_tf_avg_time(line))
print("{:20s}{:12s}{:12s}{:12s}{:12s}".format("Rammer+TVM", 
            str(rammer_autotvm_time[0]), str(rammer_autotvm_time[1]), str(rammer_autotvm_time[2]), str(rammer_autotvm_time[3])))

# TODO rammer + ansor time
rammer_ansor_time = []
for model in model_name_list_ansor_autotvm:
    with open('logs/run_nnfusion_{}_ansor.log'.format(model), 'r') as f:
        for line in f:
            if 'Summary: [min, max, mean]' in line:
                rammer_ansor_time.append(get_tf_avg_time(line))
print("{:20s}{:12s}{:12s}{:12s}{:12s}".format("Rammer+Ansor", 
            str(rammer_ansor_time[0]), str(rammer_ansor_time[1]), str(rammer_ansor_time[2]), str(rammer_ansor_time[3])))

# rammer + roller time
rammer_roller_time = []
for model in model_name_list:
    with open('logs/{}.rammer_roller.1000.log'.format(model), 'r') as f:
        for line in f:
            if 'Summary: [min, max, mean]' in line:
                rammer_roller_time.append(get_tf_avg_time(line))
print("{:20s}{:12s}{:12s}{:12s}{:12s}".format("Rammer+Roller", 
            str(rammer_roller_time[0]), str(rammer_roller_time[1]), str(rammer_roller_time[2]), str(rammer_roller_time[3])))

print("--------------------------------------------------------------\n")

# TODO ansor compile time
ansor_compile_time = []
for model in model_name_list_ansor_autotvm:
    with open('logs/compile_time_{}_ansor.log'.format(model), 'r') as f:
        compile_time = []
        lines = f.readlines()
        for line in lines:
            if "compilation time: " in line:
                compile_time.append(float(line.rstrip().split()[-1]))
        ansor_compile_time.append(sum(compile_time))

print("{:20s}{:12s}{:12s}{:12s}{:12s}".format("Ansor compile-time", 
            str(ansor_compile_time[0]), str(ansor_compile_time[1]), str(ansor_compile_time[2]), str(ansor_compile_time[3])))

roller compile time
roller_compile_time = []
for model in model_name_list:
    with open('logs/compile_time.{}.roller.log'.format(model), 'r') as f:
        for line in f:
            roller_compile_time.append(int(line))
print("{:20s}{:12s}{:12s}{:12s}{:12s}".format("Roller compile-time", 
            str(roller_compile_time[0] / 3600), str(roller_compile_time[1]/ 3600), str(roller_compile_time[2]/ 3600), str(roller_compile_time[3]/ 3600)))
