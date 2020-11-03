import numpy as np
import time

OUTPUT_FOLDER = './reproduce_result/'


def get_time_from_log(log_path):
    fin = open(log_path, 'r')
    lines = fin.readlines()
    fin.close()
    return lines[-1].split(' ')[-2].rstrip(']')


models = ['resnext', 'lstm']
batches = ['1', '4']
model_name_dict = {'resnext': 'resnext_nchw', 'lstm': 'lstm'}
model_name_dict2 = {'resnext': 'ResNeXt', 'lstm': 'LSTM'}
baselines = ['rammerbase', 'rammer']
mode = ['fast', 'select']
num_iter = 1000

reproduce_result = {}

# for model in models:
#     # extract data
#     result_record = {}
#     for batch in batches:
#         batch_result_record = {}
#         for baseline in baselines:
#             log_path = "logs/" + model_name_dict[model] + "_bs" + batch + "." + baseline + "." + str(num_iter) + ".log"
#             eval_time = get_time_from_log(log_path)
#             batch_result_record.update({baseline_name_dict[baseline]: eval_time})
#         result_record.update({batch: batch_result_record})
#     reproduce_result.update({model_name_dict[model]: result_record})


#     # save dat
#     output_dat_path = OUTPUT_FOLDER + "gpu1_batch_cuda_" + model + ".dat"
#     fout = open(output_dat_path, 'w')
#     fout.write(model_name_dict2[model]+"\tTF-1.15.2\tTF-XLA-1.15.2\tTF-TRT-7.0\tTVM-0.7\tRammer-Base\tRammer\n")
#     fout.write("BS1\t" + result_record['1'][baseline_name_dict['tf']] + "\t" + result_record['1'][baseline_name_dict['xla']] + "\t" + result_record['1'][baseline_name_dict['trt']] + "\t" + result_record['1'][baseline_name_dict['tvm']] + "\t" + result_record['1'][baseline_name_dict['rammerbase']] + "\t" + result_record['1'][baseline_name_dict['rammer']] + "\n")
#     fout.write("BS4\t" + result_record['4'][baseline_name_dict['tf']] + "\t" + result_record['4'][baseline_name_dict['xla']] + "\t" + result_record['4'][baseline_name_dict['trt']] + "\t" + result_record['4'][baseline_name_dict['tvm']] + "\t" + result_record['4'][baseline_name_dict['rammerbase']] + "\t" + result_record['4'][baseline_name_dict['rammer']] + "\n")
#     fout.write("BS16\t" + result_record['16'][baseline_name_dict['tf']] + "\t" + result_record['16'][baseline_name_dict['xla']] + "\t" + result_record['16'][baseline_name_dict['trt']] + "\t" + result_record['16'][baseline_name_dict['tvm']] + "\t" + result_record['16'][baseline_name_dict['rammerbase']] + "\t" + result_record['16'][baseline_name_dict['rammer']] + "\n")


output_dat_path = OUTPUT_FOLDER + "gpu1_batch_cuda_resnext2.dat"
fout = open(output_dat_path, 'w')
fout.write('ResNeXt-29	Rammer-Base-Fast	Rammer-Fast	Rammer-Base	Rammer\n')
fout.write('BS1\t' + str(get_time_from_log('logs/resnext_nchw_bs1.rammerbase.fast.1000.log')) + '\t' + str(get_time_from_log('logs/resnext_nchw_bs1.rammer.fast.1000.log')) + '\t' + str(get_time_from_log('logs/resnext_nchw_bs1.rammerbase.select.1000.log')) + '\t' + str(get_time_from_log('logs/resnext_nchw_bs1.rammer.select.1000.log')) + '\n')
fout.write('BS4\t' + str(get_time_from_log('logs/resnext_nchw_bs4.rammerbase.fast.1000.log')) + '\t' + str(get_time_from_log('logs/resnext_nchw_bs4.rammer.fast.1000.log')) + '\t' + str(get_time_from_log('logs/resnext_nchw_bs4.rammerbase.select.1000.log')) + '\t' + str(get_time_from_log('logs/resnext_nchw_bs4.rammer.select.1000.log')) + '\n')

output_dat_path = OUTPUT_FOLDER + "gpu1_batch_cuda_lstm2.dat"
fout = open(output_dat_path, 'w')
fout.write('LSTM	Rammer-Base-Fast	Rammer-Fast	Rammer-Base	Rammer\n')
fout.write('BS1\t' + str(get_time_from_log('logs/lstm_bs1.rammerbase.fast.1000.log')) + '\t' + str(get_time_from_log('logs/lstm_bs1.rammer.fast.1000.log')) + '\t' + str(get_time_from_log('logs/lstm_bs1.rammerbase.select.1000.log')) + '\t' + str(get_time_from_log('logs/lstm_bs1.rammer.select.1000.log')) + '\n')
fout.write('BS4\t' + str(get_time_from_log('logs/lstm_bs4.rammerbase.fast.1000.log')) + '\t' + str(get_time_from_log('logs/lstm_bs4.rammer.fast.1000.log')) + '\t' + str(get_time_from_log('logs/lstm_bs4.rammerbase.select.1000.log')) + '\t' + str(get_time_from_log('logs/lstm_bs4.rammer.select.1000.log')) + '\n')