# {"i": ["cuda -model=unknown", "topi_nn_depthwise_conv2d_nchw", [["TENSOR", [1, 512, 14, 14], "float32"], ["TENSOR", [512, 1, 3, 3], "float32"], [2, 2], [1, 1, 1, 1], [1, 1], "float32"], {}, ["depthwise_conv2d_nchw", [1, 512, 14, 14, "float32"], [512, 1, 3, 3, "float32"], [2, 2], [1, 1, 1, 1], [1, 1], "float32"], {"i": 484, "t": "direct", "c": null, "e": [["tile_f", "sp", [-1, 16, 32, 1]], ["tile_y", "sp", [-1, 1, 7, 1]], ["tile_x", "sp", [-1, 1, 1, 1]], ["auto_unroll_max_step", "ot", 0], ["unroll_explicit", "ot", 0]]}], "r": [[1000000000.0], 1, 0.029305219650268555, 1598686547.216569], "v": 0.1}
# {"i": ["cuda", "tvm_depthwise_conv2d_nchw_tune_op", [[1, 128, 16, 16], [128, 1, 5, 5], [2, 2], [1, 1, 2, 2], [1, 1]], {}, null, {"i": 157421, "t": "direct", "c": null, "e": [["tile_f", "sp", [-1, 2, 1, 16]], ["tile_y", "sp", [-1, 2, 1, 2]], ["tile_x", "sp", [-1, 2, 2, 1]], ["auto_unroll_max_step", "ot", 0], ["unroll_explicit", "ot", 1]]}], "r": [[0.0004185674826589595], 0, 2.0127649307250977, 1598630702.768298], "v": 0.1}

import json
import sys
import tensorflow as tf

flags = tf.flags
flags.DEFINE_string("input_path", "", "path of input file")
flags.DEFINE_string("output_path", "", "path of output file")

FLAGS = flags.FLAGS

def convert_depthwise_conv2d_from_kernel_to_relay_format(kernel_log_entry):
    kernel_format = json.loads(kernel_log_entry)
    # print(kernel_format)
    relay_format = {
        "i": ["cuda -model=unknown", "topi_nn_depthwise_conv2d_nchw", [["TENSOR", kernel_format['i'][2][0], "float32"], ["TENSOR", kernel_format['i'][2][1], "float32"], kernel_format['i'][2][2], kernel_format['i'][2][3], kernel_format['i'][2][4], "float32"], {}, ["depthwise_conv2d_nchw", kernel_format['i'][2][0]+["float32"], kernel_format['i'][2][1]+["float32"], kernel_format['i'][2][2], kernel_format['i'][2][3], kernel_format['i'][2][4], "float32"], {"i": kernel_format['i'][5]['i'], "t": "direct", "c": "null", "e": kernel_format['i'][5]['e']}],
        "r": kernel_format['r'],
        "v": 0.1
    }
    return relay_format

lines = open(FLAGS.input_path).readlines()

fout = open(FLAGS.output_path, "w")

for line in lines:
    relay_entry = convert_depthwise_conv2d_from_kernel_to_relay_format(line.rstrip('\n'))
    # fout.write(relay_entry+"\n")
    json.dump(relay_entry, fout)
    fout.write("\n")