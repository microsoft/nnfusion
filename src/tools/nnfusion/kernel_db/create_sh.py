import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_file', required=True, type=str, default='')
parser.add_argument('--dst_file', required=True, type=str, default='')
args = parser.parse_args()

src_file = args.src_file
dst_file = args.dst_file

with open(dst_file, 'w') as fw:
    with open(src_file, 'r') as f:
        for line in f.readlines():
            cmd = ""
            op_type, input, output, fused_nodes, strides, pad, dilation_window, axis, format = line.split('\t')
            inputs = input.split('Shape')
            if op_type == 'Convolution':
                if len(inputs) > 3:
                    op_type = 'Fused_Convolution_Add'
                    if fused_nodes == 'relu':
                        op_type = 'Fused_Convolution_Add_Relu'
                elif fused_nodes == 'relu':
                    op_type = 'Fused_Convolution_Relu'
                cmd += 'python parse_code.py --op_type ' + op_type + ' '
                input0 = inputs[1][1:-1].replace(',', '')
                input1 = inputs[2][1:-1].replace(',', '')
                cmd += '--input0_shape ' + input0 + ' '
                cmd += '--input1_shape ' + input1 + ' '
                cmd += '--output0_shape ' + output[6:-1].replace(',', '') + ' '

                cmd += '--stride ' + strides[8:-1].replace(',','') + ' '
                cmd += '--padding ' + pad[15: 19].replace(',','') + ' '
                cmd += '--dilation ' + dilation_window[8: -1].replace(',', '') + ' '
                cmd += '\n'
            elif op_type == 'DepthwiseConv2dNative':
                cmd += 'python parse_code.py --op_type ' + op_type + ' '
                input0 = inputs[1][1:-1].replace(',', '')
                input1 = inputs[2][1:-1].replace(',', '')
                cmd += '--input0_shape ' + input0 + ' '
                cmd += '--input1_shape ' + input1 + ' '
                cmd += '--output0_shape ' + output[6:-1].replace(',', '') + ' '

                cmd += '--stride ' + strides[1:-1].replace(',',' ') + ' '
                cmd += '--padding ' + pad[1: 4].replace(',',' ') + ' '
                cmd += '--dilation ' + dilation_window[1: -1].replace(',', ' ') + ' '
                cmd += '\n'
            elif op_type == 'MaxPool' or op_type == 'AvgPool':
                cmd += 'python parse_code.py --op_type ' + op_type + ' '
                input0 = inputs[1][1:-1].replace(',', '')
                cmd += '--input0_shape ' + input0 + ' '
                cmd += '--output0_shape ' + output[6:-1].replace(',', '') + ' '
                cmd += '--window_shape ' + dilation_window[6:-1].replace(',', '') + ' '
                cmd += '--stride ' + strides[8:-1].replace(',','') + ' '
                cmd += '--padding ' + pad[6: 10].replace(',','') + ' '
                cmd += '\n'
            elif op_type == "Sum":
                cmd += 'python parse_code.py --op_type ' + op_type + ' '
                input0 = inputs[1][1:-1].replace(',', '')
                cmd += '--input0_shape ' + input0 + ' '
                cmd += '--output0_shape ' + output[6:-1].replace(',', '') + ' '
                cmd += '--reduction_axis ' + axis[8:-1].replace(',', '') + ' '
                cmd += '\n'
            elif op_type == "Dot" or op_type == "BatchMatMul":
                cmd += 'python parse_code.py --op_type ' + op_type + ' '
                input0 = inputs[1][1:-1].replace(',', '')
                input1 = inputs[2][1:-1].replace(',', '')
                cmd += '--input0_shape ' + input0 + ' '
                cmd += '--input1_shape ' + input1 + ' '
                cmd += '--output0_shape ' + output[6:-1].replace(',', '') + ' '
                cmd += '\n'
            fw.write(cmd)
            
            

            
