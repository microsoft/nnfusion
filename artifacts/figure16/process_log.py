# import tensorflow as tf
import numpy as np
import time

# flags = tf.flags
# logging = tf.logging
# logging.set_verbosity(tf.logging.ERROR)

# flags.DEFINE_string('iter_file', 'logs/alexnet_nchw_xla.iter_time.bs_1.100+5.log', 'file name of the tf log file')
# flags.DEFINE_string('nvprof_file', 'logs/alexnet_nchw_xla.nvprof.bs_1.100+5.log', 'file name of the tf log file')

# FLAGS = flags.FLAGS

def process_log(iter_file, nvprof_file):
    num_warmup = int(nvprof_file.split('.')[-2].split('+')[1])
    num_step = int(nvprof_file.split('.')[-2].split('+')[0])
    num_iter = num_step + num_warmup

    fin = open(nvprof_file, 'r')
    lines = fin.readlines()
    fin.close()
    start_point = -1
    for i in range(len(lines)):
        if "\"Avg\",\"Min\",\"Max\",\"Name\"" in lines[i]:
            start_point = i + 1
            break
    assert(start_point != -1)
    line = lines[start_point + 1]
    items = line.split(',')
    kernel_port = float(items[1])
    kernel_time = float(items[2])
    time_unit = lines[start_point].split(',')[2]
    fix = 1.0
    if time_unit == 'us':
        fix = 0.001
    if time_unit == 's':
        fix = 1000.0
    all_kernel_time = kernel_time * fix / kernel_port * 100.0 / num_iter


    fin = open(iter_file, 'r')
    lines = fin.readlines()
    fin.close()
    line = lines[-1]
    items = line.split(',')
    all_run_time = float(items[-1].rstrip('] ms\n'))

    overhead = all_run_time - all_kernel_time

    # print(all_run_time, all_kernel_time, all_kernel_time / all_run_time)

    return all_run_time, all_kernel_time, overhead, all_kernel_time / all_run_time, overhead / all_run_time



models = ['resnext_nchw', 'nasnet_cifar_nchw',
          'alexnet_nchw', 'deepspeech2', 'lstm', 'seq2seq']
model_name_dict2 = {'resnext_nchw': 'ResNeXt', 'nasnet_cifar_nchw': 'NASNet',
                   'alexnet_nchw': 'AlexNet', 'deepspeech2': 'DeepSpeech2', 'lstm': 'LSTM', 'seq2seq': 'Seq2Seq'}
baselines = ['tf', 'rammerbase', 'rammer']
# baselines = ['tf']
batches = [1]
prefix = 'logs/'
warmup = 5
steps = 1000


output_dat_path = './reproduce_result/gpu1_gpu_schedoverhead_cuda.dat'
fout = open(output_dat_path, 'w')
fout.write("Baseline	RexNeXt-Kernel	RexNeXt-Overhead	RexNeXt-Proportion	NASNet-Kernel	NASNet-Overhead	NASNet-Proportion	AlexNet-Kernel	AlexNet-Overhead	AlexNet-Proportion	DeepSpeech2-Kernel	DeepSpeech2-Overhead	DeepSpeech2-Proportion	LSTM-Kernel	LSTM-Overhead	LSTM-Proportion	Seq2Seq-Kernel	Seq2Seq-Overhead	Seq2Seq-Proportion	Text-Height-Offset\n")

baseline = 'tf'
data = []
for model in models:
    # print(model)
    record = {}
    iter_file = prefix + model + '_bs' + str(1) + '.' + baseline + '.iter_time' + '.' + str(steps) + '+' + str(warmup) + '.log'
    nvprof_file = prefix + model + '_bs' + str(1) + '.' + baseline + '.nvprof' + '.' + str(steps) + '+' + str(warmup) + '.log'
    result = process_log(iter_file, nvprof_file)
        # print(result[0], result[1], result[2], result[3], result[4])
    data = data + [result[1], result[2], result[4] * 100.0]
fout.write("TF")
for item in data:
    fout.write("\t" + str(item))
fout.write("\t8\n")


baseline = 'rammerbase'
data = []
for model in models:
    # print(model)
    record = {}
    iter_file = prefix + model + '_bs' + str(1) + '.' + baseline + '.iter_time' + '.' + str(steps) + '+' + str(warmup) + '.log'
    nvprof_file = prefix + model + '_bs' + str(1) + '.' + baseline + '.nvprof' + '.' + str(steps) + '+' + str(warmup) + '.log'
    result = process_log(iter_file, nvprof_file)
        # print(result[0], result[1], result[2], result[3], result[4])
    data = data + [result[1], result[2], result[4] * 100.0]
fout.write("RB")
for item in data:
    fout.write("\t" + str(item))
fout.write("\t9\n")


baseline = 'rammer'
data = []
for model in models:
    # print(model)
    record = {}
    iter_file = prefix + model + '_bs' + str(1) + '.' + baseline + '.iter_time' + '.' + str(steps) + '+' + str(warmup) + '.log'
    nvprof_file = prefix + model + '_bs' + str(1) + '.' + baseline + '.nvprof' + '.' + str(steps) + '+' + str(warmup) + '.log'
    result = process_log(iter_file, nvprof_file)
        # print(result[0], result[1], result[2], result[3], result[4])
    data = data + [result[1], result[2], result[4] * 100.0]
fout.write("R")
for item in data:
    fout.write("\t" + str(item))
fout.write("\t-1\n")