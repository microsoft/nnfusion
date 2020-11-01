# {"i": ["cuda", "tvm_matmul_tune_op", [1, 256, 10], {}, ["tvm_matmul_tune_op", 1, 256, 10], {"i": 6, "t": "", "c": null, "e": [["tile_k", "sp", [-1, 64]]]}], "r": [[7.258520111880661e-06], 0, 0.9687888622283936, 1598593289.460491], "v": 0.1}
# {"i": ["cuda -model=unknown", "topi_nn_dense", [["TENSOR", [1, 512], "float32"], ["TENSOR", [1000, 512], "float32"], null, "float32"], {}, ["dense", [1, 512, "float32"], [1000, 512, "float32"], 0, "float32"], {"i": 3, "t": "direct", "c": null, "e": [["tile_k", "sp", [-1, 8]]]}], "r": [[6.859512920731313e-06], 0, 1.7214715480804443, 1598530753.2381613], "v": 0.1}

import json
import sys
import tensorflow as tf

flags = tf.flags
flags.DEFINE_string("input_path", "", "path of input file")
flags.DEFINE_string("output_path", "", "path of output file")

FLAGS = flags.FLAGS

def convert_dense_from_kernel_to_relay_format(kernel_log_entry):
    kernel_format = json.loads(kernel_log_entry)
    # print(kernel_format)
    relay_format = {
        "i": ["cuda -model=unknown", "topi_nn_dense", [["TENSOR", [kernel_format['i'][2][0], kernel_format['i'][2][1]], "float32"], ["TENSOR", [kernel_format['i'][2][2], kernel_format['i'][2][1]], "float32"], "null", "float32"], {}, ["dense", [kernel_format['i'][2][0], kernel_format['i'][2][1], "float32"], [kernel_format['i'][2][2], kernel_format['i'][2][1], "float32"], 0, "float32"], {"i": kernel_format['i'][5]['i'], "t": "direct", "c": "null", "e": kernel_format['i'][5]['e']}],
        "r": kernel_format['r'],
        "v": 0.1
    }
    return relay_format

lines = open(FLAGS.input_path).readlines()

fout = open(FLAGS.output_path, "w")

for line in lines:
    relay_entry = convert_dense_from_kernel_to_relay_format(line.rstrip('\n'))
    # fout.write(relay_entry+"\n")
    json.dump(relay_entry, fout)
    fout.write("\n")