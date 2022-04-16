import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str,
                    default='../logs_v2_rocm/roller/wo_storage_align/pooling')
parser.add_argument('--op', type=str, default='pooling')

args = parser.parse_args()

files = os.listdir(args.log_dir)

# print(files)


def get_file_name(op_type, id):
    global files
    prefix = op_type+str(id)
    for file in files:
        if prefix == file.split('_')[0]:
            return file
    return None


for i in range(1000):
    file_name = get_file_name(args.op, i)
    if file_name == None:
        exit(0)
    file = open(args.log_dir + "/" + file_name)
    lines = file.readlines()
    file.close()
    results = ['inf', 'inf', 'inf']
    cnt = 0
    for line in lines:
        if "best time: " in line:
            results[cnt] = line.rstrip('ms\n').split(' ')[-1]
            cnt += 1
    if cnt < 2:
        results[1] = results[0]
    if cnt < 3:
        results[2] = results[1]
    print(results[0]+'\t'+results[2])

