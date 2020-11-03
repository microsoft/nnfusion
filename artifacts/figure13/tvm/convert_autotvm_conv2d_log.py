import json
import sys
import tensorflow as tf

flags = tf.flags
flags.DEFINE_string("input_path", "", "path of input file")
flags.DEFINE_string("output_path", "", "path of output file")

FLAGS = flags.FLAGS

def convert_conv2d_from_kernel_to_relay_format(kernel_log_entry):
    kernel_format = json.loads(kernel_log_entry)
    # print(kernel_format)
    relay_format = {
        "i": ["cuda -model=unknown", "topi_nn_conv2d", [["TENSOR", kernel_format['i'][2][0], "float32"], ["TENSOR", kernel_format['i'][2][1], "float32"], kernel_format['i'][2][3], [kernel_format['i'][2][4][0], kernel_format['i'][2][4][0], kernel_format['i'][2][4][1], kernel_format['i'][2][4][1]], kernel_format['i'][2][5], "NCHW", "float32"], {}, ["conv2d", kernel_format['i'][2][0]+["float32"], kernel_format['i'][2][1]+["float32"], kernel_format['i'][2][3], [kernel_format['i'][2][4][0], kernel_format['i'][2][4][0], kernel_format['i'][2][4][1], kernel_format['i'][2][4][1]], kernel_format['i'][2][5], "NCHW", "float32"], {"i": kernel_format['i'][5]['i'], "t": "direct", "c": "null", "e": kernel_format['i'][5]['e']}],
        "r": kernel_format['r'],
        "v": 0.1
    }
    return relay_format

lines = open(FLAGS.input_path).readlines()

fout = open(FLAGS.output_path, "w")

for line in lines:
    relay_entry = convert_conv2d_from_kernel_to_relay_format(line.rstrip('\n'))
    # fout.write(relay_entry+"\n")
    json.dump(relay_entry, fout)
    fout.write("\n")