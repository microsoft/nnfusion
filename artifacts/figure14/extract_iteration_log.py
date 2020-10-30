import tensorflow as tf
import numpy as np
import time

# flags = tf.flags
# logging = tf.logging
# logging.set_verbosity(tf.logging.ERROR)

# flags.DEFINE_string('kernel_trace_file', '', 'file name of the tf kernel trace file')
# flags.DEFINE_string('kernel_metric_file', '', 'file name of the tf nvprof metric file')

# FLAGS = flags.FLAGS


def extract_iteration_kernel_trace_tf(log_file, head_len=1, tail_len=1):
    lines = open(log_file).readlines()
    start_point = -1
    for i in range(len(lines)-2, -1, -1):
        # print(lines[i])
        if '[CUDA memcpy DtoH]' in lines[i]:
            start_point = i + 1
            break
    assert (start_point != -1)
    # print(start_point)

    extracted = []
    for i in range(start_point, len(lines)):
        if '[CUDA' not in lines[i]:
            extracted.append(lines[i])

    fout = open(log_file.rstrip('.log') + ".extracted.log", 'w')
    for line in extracted:
        fout.write(line)
    fout.close()

    head_kernels = []
    for i in range(start_point, len(lines)):
        if '[CUDA' not in lines[i]:
            kernel_name = lines[i].rstrip('\n').split('\",')[-2].lstrip('\"')
            head_kernels.append(kernel_name)
        if len(head_kernels) >= head_len:
            break

    tail_kernels = []
    for i in range(len(lines) - 1, -1, -1):
        if '[CUDA' not in lines[i]:
            kernel_name = lines[i].rstrip('\n').split('\",')[-2].lstrip('\"')
            tail_kernels.append(kernel_name)
        if len(tail_kernels) >= tail_len:
            break

    return (len(extracted), head_kernels, tail_kernels)


def extract_iteration_kernel_metric_tf(log_file, len_kernel_trace, head_kernels, tail_kernels):
    # print(head_kernels)
    # print(tail_kernels)
    lines = open(log_file).readlines()
    extracted = []
    if ('lstm' in log_file) or ('seq2seq' in log_file):
        for i in range(len(lines) - len_kernel_trace, len(lines)):
            extracted.append(lines[i])
    else:
        start_point = -1
        for i in range(len(lines) - 1 - len(head_kernels), -1 + len(tail_kernels), -1):
            # check head
            flag_head_matched = True
            for j in range(len(head_kernels)):
                kernel_name = lines[i + j].rstrip('\n').split('\",')[-2].lstrip('\"')
                if kernel_name != head_kernels[j]:
                    flag_head_matched = False
            flag_tail_matched = True
            for j in range(len(tail_kernels)):
                kernel_name = lines[i - j - 1].rstrip('\n').split('\",')[-2].lstrip('\"')
                if kernel_name != tail_kernels[j]:
                    flag_tail_matched = False
            if flag_head_matched and flag_tail_matched:
                start_point = i
                break
        assert (start_point != -1)
        # print(start_point)
        for i in range(start_point, len(lines)):
            extracted.append(lines[i])
    fout = open(log_file.rstrip('.log') + ".extracted.log", 'w')
    for line in extracted:
        fout.write(line)
    fout.close()


def extract_iteration_kernel_trace_trt_native(log_file, head_len=1, tail_len=1):
    lines = open(log_file).readlines()
    start_point = -1
    for i in range(len(lines)-2, -1, -1):
        # print(lines[i])
        if '[CUDA memcpy DtoH]' in lines[i]:
            start_point = i + 1
            break
    assert (start_point != -1)
    # print(start_point)

    extracted = []
    for i in range(start_point, len(lines)):
        if '[CUDA' not in lines[i]:
            extracted.append(lines[i])

    fout = open(log_file.rstrip('.log') + ".extracted.log", 'w')
    for line in extracted:
        fout.write(line)
    fout.close()

    head_kernels = []
    for i in range(start_point, len(lines)):
        if '[CUDA' not in lines[i]:
            kernel_name = lines[i].rstrip('\n').split('\",')[-2].lstrip('\"')
            head_kernels.append(kernel_name)
        if len(head_kernels) >= head_len:
            break

    tail_kernels = []
    for i in range(len(lines) - 1, -1, -1):
        if '[CUDA' not in lines[i]:
            kernel_name = lines[i].rstrip('\n').split('\",')[-2].lstrip('\"')
            tail_kernels.append(kernel_name)
        if len(tail_kernels) >= tail_len:
            break

    return (len(extracted), head_kernels, tail_kernels)


def extract_iteration_kernel_metric_trt_native(log_file, len_kernel_trace, head_kernels, tail_kernels):
    # print(head_kernels)
    # print(tail_kernels)
    lines = open(log_file).readlines()
    extracted = []
    if ('lstm' in log_file) or ('seq2seq' in log_file):
        for i in range(len(lines) - len_kernel_trace, len(lines)):
            extracted.append(lines[i])
    else:
        start_point = -1
        for i in range(len(lines) - 1 - len(head_kernels), -1 + len(tail_kernels), -1):
            # check head
            flag_head_matched = True
            for j in range(len(head_kernels)):
                kernel_name = lines[i + j].rstrip('\n').split('\",')[-2].lstrip('\"')
                if kernel_name != head_kernels[j]:
                    flag_head_matched = False
            flag_tail_matched = True
            for j in range(len(tail_kernels)):
                kernel_name = lines[i - j - 1].rstrip('\n').split('\",')[-2].lstrip('\"')
                if kernel_name != tail_kernels[j]:
                    flag_tail_matched = False
            if flag_head_matched and flag_tail_matched:
                start_point = i
                break
        assert (start_point != -1)
        # print(start_point)
        for i in range(start_point, len(lines)):
            extracted.append(lines[i])
    fout = open(log_file.rstrip('.log') + ".extracted.log", 'w')
    for line in extracted:
        fout.write(line)
    fout.close()


def extract_iteration_kernel_trace_rammer(log_file):
    lines = open(log_file).readlines()
    start_point = -1
    for i in range(len(lines)):
        if ',,,,,,' in lines[i]:
            start_point = i + 1
            break
    assert (start_point != -1)

    extracted = []
    for i in range(start_point, len(lines)):
        if '[CUDA' not in lines[i]:
            extracted.append(lines[i])
    fout = open(log_file.rstrip('.log') + ".extracted.log", 'w')
    for line in extracted:
        fout.write(line)
    fout.close()


def extract_iteration_kernel_metric_rammer(log_file):
    lines = open(log_file).readlines()
    start_point = -1
    for i in range(len(lines)):
        if lines[i] == ',,,,,%\n':
            start_point = i + 1
            break
    assert (start_point != -1)

    extracted = []
    for i in range(start_point, len(lines)):
        extracted.append(lines[i])
    fout = open(log_file.rstrip('.log') + ".extracted.log", 'w')
    for line in extracted:
        fout.write(line)
    fout.close()

# extract_iteration_kernel_trace(FLAGS.kernel_trace_file)

models = ['resnext_nchw', 'nasnet_cifar_nchw', 'alexnet_nchw', 'deepspeech2', 'lstm', 'seq2seq']
baselines = ['tf', 'trt', 'rammerbase', 'rammer']
batches = [1]
prefix = 'logs/'
warmup = 1
steps = 3

for model in models:
    for baseline in baselines:
        if baseline == 'tf':
            for batch in batches:
                kernel_trace_file = prefix + model + '_bs' + \
                    str(batch) + '.tf.kernel_trace.' + \
                    str(steps) + '+' + str(warmup) + '.log'
                kernel_metric_file = prefix + model + '_bs' + \
                    str(batch) + '.tf.sm_efficiency.' + \
                    str(steps) + '+' + str(warmup) + '.log'
                results = extract_iteration_kernel_trace_tf(kernel_trace_file)
                extract_iteration_kernel_metric_tf(kernel_metric_file, results[0], results[1], results[2])
        elif baseline == 'trt':
            for batch in batches:
                kernel_trace_file = prefix + model + '_bs' + \
                    str(batch) + '.trt.kernel_trace.' + \
                    str(steps) + '+' + str(warmup) + '.log'
                kernel_metric_file = prefix + model + '_bs' + \
                    str(batch) + '.trt.sm_efficiency.' + \
                    str(steps) + '+' + str(warmup) + '.log'
                results = extract_iteration_kernel_trace_tf(kernel_trace_file)
                extract_iteration_kernel_metric_tf(kernel_metric_file, results[0], results[1], results[2])
        else:
            for batch in batches:
                kernel_trace_file = prefix + model + '_bs' + \
                    str(batch) + '.' + baseline + '.kernel_trace.log'
                kernel_metric_file = prefix + model + '_bs' + \
                    str(batch) + '.' + baseline + '.sm_efficiency.log'
                extract_iteration_kernel_trace_rammer(kernel_trace_file)
                extract_iteration_kernel_metric_rammer(kernel_metric_file)
