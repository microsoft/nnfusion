

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
    if 'bert' in model or 'nasnet' in model:
        xla_time.append('OOM')
        continue
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

# ansor time
ansor_time = []
for model in model_name_list:
    try:
        with open('logs/ansor_{}.log'.format(model), 'r') as f:
            for line in f:
                if line.startswith('Summary:'):
                    ansor_time.append(float(line.split(',')[1]) * 1000)
    except:
        ansor_time.append(-1)
print("{:20s}{:12s}{:12s}{:12s}{:12s}".format("Ansor",
            "N/A" if ansor_time[0] < 0 else str(ansor_time[0])[:12], str(ansor_time[1])[:10], str(ansor_time[2])[:10], str(ansor_time[3])[:10]))

# rammer + tvm time
rammer_autotvm_time = []
for model in model_name_list_ansor_autotvm:
    with open('logs/run_nnfusion_{}_autotvm.log'.format(model), 'r') as f:
        for line in f:
            if 'Summary: [min, max, mean]' in line:
                rammer_autotvm_time.append(get_tf_avg_time(line))
print("{:20s}{:12s}{:12s}{:12s}{:12s}".format("Rammer+TVM", 
            str(rammer_autotvm_time[0])[:11], str(rammer_autotvm_time[1]), str(rammer_autotvm_time[2]), str(rammer_autotvm_time[3])))

# rammer + ansor time
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

# ansor compile time
import json
def get_ansor_e2e_log(filename):
    compile_time = 0.0
    with open(filename, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            obj = json.loads(line)
            if "r" in obj:
                result = obj["r"]
            else:
                result = obj["result"]
            # result[0] runtime
            # result[1] error code
            # result[2] compilation time
            if result[1] == 0:
                compile_time += result[2]
        return compile_time
ansor_compile_time = []
ansor_compile_time.append(get_ansor_e2e_log('logs/frozen_bert_large_infer_bs128.pb.autotvm_tuned_1000.67.log'))
ansor_compile_time[0] += get_ansor_e2e_log('logs/matmul_65536_1024_1024.log')
ansor_compile_time[0] += get_ansor_e2e_log('logs/matmul_65536_1024_4096.log')
ansor_compile_time[0] += get_ansor_e2e_log('logs/matmul_65536_2_1024.log')
ansor_compile_time[0] += get_ansor_e2e_log('logs/matmul_65536_30522_1024.log')
ansor_compile_time[0] += get_ansor_e2e_log('logs/matmul_65536_4096_1024.log')
ansor_compile_time.append(get_ansor_e2e_log('logs/frozen_lstm_infer_bs128.pb.ansor_tuned_20000.log'))
ansor_compile_time.append(get_ansor_e2e_log('logs/frozen_nasnet_large_nchw_infer_bs128.pb.autotvm_tuned_93000.log'))
ansor_compile_time.append(get_ansor_e2e_log('logs/frozen_resnet50_infer_bs128.pb.ansor_tuned_27000.log'))
# for model in model_name_list_ansor_autotvm:
#     with open('logs/compile_time_{}_ansor.log'.format(model), 'r') as f:
#         compile_time = []
#         lines = f.readlines()
#         for line in lines:
#             if "compilation time: " in line:
#                 compile_time.append(float(line.rstrip().split()[-1]))
#         ansor_compile_time.append(sum(compile_time))

print("{:20s}{:12s}{:12s}{:12s}{:12s}".format("Ansor compile-time", 
            str(ansor_compile_time[0] / 3600)[:4] + 'h(TVM)', str(ansor_compile_time[1] / 3600)[:4] + 'h', str(ansor_compile_time[2] / 3600)[:4] + 'h', str(ansor_compile_time[3] / 3600)[:4] + 'h'))

# roller compile time
roller_compile_time = []
for model in model_name_list:
    with open('logs/compile_time.{}.roller.log'.format(model), 'r') as f:
        for line in f:
            roller_compile_time.append(int(line))
print("{:20s}{:12s}{:12s}{:12s}{:12s}".format("Roller compile-time", 
            str(roller_compile_time[0]) + 's', str(roller_compile_time[1]) + 's', str(roller_compile_time[2]) + 's', str(roller_compile_time[3]) + 's'))
