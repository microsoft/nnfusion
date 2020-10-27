# import tensorflow as tf
import numpy as np
import time
import csv

# flags = tf.flags
# logging = tf.logging
# logging.set_verbosity(tf.logging.ERROR)

# flags.DEFINE_string('iter_file', 'logs/alexnet_nchw_xla.iter_time.bs_1.100+5.log', 'file name of the tf log file')
# flags.DEFINE_string('nvprof_file', 'logs/alexnet_nchw_xla.nvprof.bs_1.100+5.log', 'file name of the tf log file')

# FLAGS = flags.FLAGS


def process_log_rammer(kernel_trace_file, sm_efficiency_file):
    # num_warmup = int(nvprof_file.split('.')[-2].split('+')[1])
    # num_step = int(nvprof_file.split('.')[-2].split('+')[0])
    # num_iter = num_step + num_warmup

    kernel_logs = open(kernel_trace_file, 'r').readlines()
    metric_logs = open(sm_efficiency_file, 'r').readlines()

    entry_name = ["Name", "Start", "Duration", "sm_efficiency"]

    kernel_name = []
    kernel_start = []
    kernel_dur = []
    kernel_sm_efficiency = []

    for idx in range(len(metric_logs)):
        i = idx
        kernel_items = kernel_logs[i].rstrip('\n').split(',')
        metric_items = metric_logs[i].rstrip('\n').split(',')
        kernel_name.append(metric_items[0])
        kernel_start.append(float(kernel_items[0]))
        kernel_dur.append(float(kernel_items[1]))
        kernel_sm_efficiency.append(float(metric_items[-1]))

    kernel_time_sum = 0
    all_sm_efficiency = 0
    for i in range(len(kernel_name)):
        all_sm_efficiency += kernel_dur[i] * kernel_sm_efficiency[i]
        kernel_time_sum += kernel_dur[i]

    return all_sm_efficiency / kernel_time_sum


def process_log_tf(kernel_trace_file, sm_efficiency_file):
    kernel_logs = open(kernel_trace_file, 'r').readlines()
    metric_logs = open(sm_efficiency_file, 'r').readlines()

    entry_name = ["Name", "Start", "Duration", "sm_efficiency"]

    kernel_name = []
    kernel_start = []
    kernel_dur = []
    kernel_sm_efficiency = []

    j = 0

    for i in range(len(metric_logs)):
        if j >= len(kernel_logs):
            break
        kernel_log = kernel_logs[j]
        kernel_items = kernel_logs[j].rstrip('\n').split(',')
        metric_items = metric_logs[i].rstrip('\n').split(',')

        kernel_name.append(metric_items[0])
        kernel_sm_efficiency.append(float(metric_items[-1]))

        acc_start = 0
        acc_dur = 0

        if ('cudnn' in kernel_log) and not ('pooling' in kernel_log):
            acc_start = float(kernel_items[0])
            acc_dur = float(kernel_items[1])
            while True:
                j += 1
                if j >= len(kernel_logs):
                    break
                kernel_log = kernel_logs[j]
                kernel_items = kernel_logs[j].rstrip('\n').split(',')
                if ('cudnn' in kernel_log or 'gemm' in kernel_log or 'fft' in kernel_log) and not ('pooling' in kernel_log) and not ('bn_fw_inf_1C11_kernel_NCHW' in kernel_log):
                    acc_dur += float(kernel_items[1])
                else:
                    j -= 1
                    break
            kernel_start.append(acc_start)
            kernel_dur.append(acc_dur)
        else:
            kernel_start.append(float(kernel_items[0]))
            kernel_dur.append(float(kernel_items[1]))
        j += 1

    # print('correctness check:', len(metric_logs), '==' , i+1, ',', len(kernel_logs), '==' , j)

    kernel_time_sum = 0
    all_sm_efficiency = 0
    for i in range(len(kernel_name)):
        all_sm_efficiency += kernel_dur[i] * kernel_sm_efficiency[i]
        kernel_time_sum += kernel_dur[i]

    return all_sm_efficiency / kernel_time_sum


models = ['resnext_nchw', 'nasnet_cifar_nchw',
          'alexnet_nchw', 'deepspeech2', 'lstm', 'seq2seq']
model_name_dict2 = {'resnext_nchw': 'ResNeXt', 'nasnet_cifar_nchw': 'NASNet',
                   'alexnet_nchw': 'AlexNet', 'deepspeech2': 'DeepSpeech2', 'lstm': 'LSTM', 'seq2seq': 'Seq2Seq'}
baselines = ['tf', 'rammerbase', 'rammer']
# baselines = ['tf']
batches = [1]
prefix = 'logs/'
warmup = 0
steps = 3

reproduced_results = {}
for model in models:
    # print(model)
    model_record = {}
    for baseline in baselines:
        if baseline == 'tf':
            # for batch in batches:
            batch = 1
            kernel_trace_file = prefix + model + '_bs' + \
                str(batch) + '.tf.kernel_trace.' + \
                str(steps) + '+' + str(warmup) + '.extracted.log'
            kernel_metric_file = prefix + model + '_bs' + \
                str(batch) + '.tf.sm_efficiency.' + \
                str(steps) + '+' + str(warmup) + '.extracted.log'
            result = process_log_tf(kernel_trace_file, kernel_metric_file)
        else:
            # for batch in batches:
            batch = 1
            kernel_trace_file = prefix + model + '_bs' + \
                str(batch) + '.' + baseline + \
                '.kernel_trace.extracted.log'
            kernel_metric_file = prefix + model + '_bs' + \
                str(batch) + '.' + baseline + \
                '.sm_efficiency.extracted.log'
            result = process_log_rammer(kernel_trace_file, kernel_metric_file)
        model_record.update({baseline: str(result)})
    reproduced_results.update({model: model_record})

# save dat
output_dat_path = './reproduce_result/gpu1_gpu_util_cuda.dat'
fout = open(output_dat_path, 'w')
fout.write("#	TF	RammerBase	Rammer\n")
for model in models:
    fout.write(model_name_dict2[model] + "\t" + str(reproduced_results[model]['tf']) + "\t" + str(reproduced_results[model]['rammerbase']) + "\t" + str(reproduced_results[model]['rammer']) + "\n")
