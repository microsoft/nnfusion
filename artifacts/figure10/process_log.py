import numpy as np
import time

OUTPUT_FOLDER = './reproduce_result/'


def get_time_from_log(log_path):
    fin = open(log_path, 'r')
    lines = fin.readlines()
    fin.close()
    return lines[-1].split(' ')[-2].rstrip(']')


models = ['resnext_nchw_bs1', 'nasnet_cifar_nchw_bs1',
          'alexnet_nchw_bs1', 'deepspeech2_bs1', 'lstm_bs1', 'seq2seq_bs1']
model_name_dict = {'resnext_nchw_bs1': 'resnext', 'nasnet_cifar_nchw_bs1': 'nasnet',
                   'alexnet_nchw_bs1': 'alexnet', 'deepspeech2_bs1': 'deepspeech2', 'lstm_bs1': 'lstm', 'seq2seq_bs1': 'seq2seq'}
model_name_dict2 = {'resnext_nchw_bs1': 'ResNeXt', 'nasnet_cifar_nchw_bs1': 'NASNet',
                   'alexnet_nchw_bs1': 'AlexNet', 'deepspeech2_bs1': 'DeepSpeech2', 'lstm_bs1': 'LSTM', 'seq2seq_bs1': 'Seq2Seq'}
baselines = ['tf', 'xla', 'trt', 'tvm', 'rammerbase', 'rammer']
baseline_name_dict = {'tf': 'TF-1.15.2', 'xla': 'TF-XLA-1.15.2', 'trt': 'TF-TRT-7.0',
                      'tvm': 'TVM-0.7', 'rammerbase': 'Rammer-Base', 'rammer': 'Rammer'}
num_iter = 1000

reproduce_result = {}

for model in models:
    # extract data
    result_record = {}
    for baseline in baselines:
        log_path = "logs/" + model + "." + baseline + "." + str(num_iter) + ".log"
        eval_time = get_time_from_log(log_path)
        result_record.update({baseline_name_dict[baseline]: eval_time})
    reproduce_result.update({model_name_dict[model]: result_record})

    # save dat
    output_dat_path = OUTPUT_FOLDER + "gpu1_e2e_cuda_" + model_name_dict[model] + ".dat"
    fout = open(output_dat_path, 'w')
    fout.write("BS1-V100\tTF-1.15.2\tTF-XLA-1.15.2\tTF-TRT-7.0\tTVM-0.7\tRammer-Base\tRammer\n")
    fout.write(model_name_dict2[model] + "\t" + result_record[baseline_name_dict['tf']] + "\t" + result_record[baseline_name_dict['xla']] + "\t" + result_record[baseline_name_dict['trt']] + "\t" + result_record[baseline_name_dict['tvm']] + "\t" + result_record[baseline_name_dict['rammerbase']] + "\t" + result_record[baseline_name_dict['rammer']] + "\n")
