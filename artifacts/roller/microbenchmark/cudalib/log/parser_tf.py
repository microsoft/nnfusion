import os
import sys
import pandas as pd
import numpy as np

def get_id(file_name):
    return int(file_name[:-4].split('_')[-1])

folder_list = ['conv','depthwise','element','matmul','pool','reduce']
for folder in folder_list:
    file_list = os.listdir(folder)
    result = []
    for file_name in file_list:
        real_file_name = folder + '/' + file_name
        with open(real_file_name,'r') as fin:
            fin_str_list = fin.readlines()
            fin.close()
        for line in fin_str_list:
            if 'ms on avg' in line:
                result.append([get_id(file_name),float(line.split()[0])])
                break
    df = pd.DataFrame(result)
    df.to_csv(folder+'_cudalib_result.csv')
