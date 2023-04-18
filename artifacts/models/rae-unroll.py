from ast_analyzer.shape_inference.types import *
from ast_analyzer.utils.save_tensor import save_tensor_bin
from ast_analyzer import workflow_fix_flag, test_torch_eval, test_torch_train, workflow_train_recursion
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ast_analyzer.grad.impl as grad
from ast_analyzer.utils.argparser import get_parser
import os
from ast_analyzer.to_onnx import to_torch_func
parser = get_parser()

parser.add_argument('--bs', type=int, default=1)
args = parser.parse_args()

depth = 7
n = 2 ** depth - 1

class RAEUnroll(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(1024, 512)
    
    def forward(self, inp):
        output_32 = inp[32]
        output_20 = inp[20]
        output_21 = inp[21]
        output_22 = inp[22]
        output_23 = inp[23]
        output_24 = inp[24]
        output_25 = inp[25]
        output_30 = inp[30]
        output_31 = inp[31]
        output_33 = self.encoder(torch.cat((output_30, output_31)))
        output_33 = torch.tanh(output_33)
        output_26 = inp[26]
        output_27 = inp[27]
        output_28 = inp[28]
        output_29 = inp[29]
        output_34 = self.encoder(torch.cat((output_28, output_29)))
        output_34 = torch.tanh(output_34)
        output_35 = self.encoder(torch.cat((output_27, output_34)))
        output_35 = torch.tanh(output_35)
        output_36 = self.encoder(torch.cat((output_26, output_35)))
        output_36 = torch.tanh(output_36)
        output_37 = self.encoder(torch.cat((output_33, output_36)))
        output_37 = torch.tanh(output_37)
        output_38 = self.encoder(torch.cat((output_25, output_37)))
        output_38 = torch.tanh(output_38)
        output_39 = self.encoder(torch.cat((output_24, output_38)))
        output_39 = torch.tanh(output_39)
        output_40 = self.encoder(torch.cat((output_23, output_39)))
        output_40 = torch.tanh(output_40)
        output_41 = self.encoder(torch.cat((output_22, output_40)))
        output_41 = torch.tanh(output_41)
        output_42 = self.encoder(torch.cat((output_21, output_41)))
        output_42 = torch.tanh(output_42)
        output_43 = self.encoder(torch.cat((output_20, output_42)))
        output_43 = torch.tanh(output_43)
        output_44 = self.encoder(torch.cat((output_32, output_43)))
        output_44 = torch.tanh(output_44)
        output_17 = inp[17]
        output_18 = inp[18]
        output_19 = inp[19]
        output_45 = self.encoder(torch.cat((output_18, output_19)))
        output_45 = torch.tanh(output_45)
        output_46 = self.encoder(torch.cat((output_17, output_45)))
        output_46 = torch.tanh(output_46)
        output_11 = inp[11]
        output_12 = inp[12]
        output_13 = inp[13]
        output_14 = inp[14]
        output_15 = inp[15]
        output_16 = inp[16]
        output_47 = self.encoder(torch.cat((output_15, output_16)))
        output_47 = torch.tanh(output_47)
        output_48 = self.encoder(torch.cat((output_14, output_47)))
        output_48 = torch.tanh(output_48)
        output_49 = self.encoder(torch.cat((output_13, output_48)))
        output_49 = torch.tanh(output_49)
        output_50 = self.encoder(torch.cat((output_12, output_49)))
        output_50 = torch.tanh(output_50)
        output_51 = self.encoder(torch.cat((output_11, output_50)))
        output_51 = torch.tanh(output_51)
        output_52 = self.encoder(torch.cat((output_46, output_51)))
        output_52 = torch.tanh(output_52)
        output_10 = inp[10]
        output_8 = inp[8]
        output_9 = inp[9]
        output_53 = self.encoder(torch.cat((output_8, output_9)))
        output_53 = torch.tanh(output_53)
        output_5 = inp[5]
        output_6 = inp[6]
        output_7 = inp[7]
        output_54 = self.encoder(torch.cat((output_6, output_7)))
        output_54 = torch.tanh(output_54)
        output_55 = self.encoder(torch.cat((output_5, output_54)))
        output_55 = torch.tanh(output_55)
        output_56 = self.encoder(torch.cat((output_53, output_55)))
        output_56 = torch.tanh(output_56)
        output_57 = self.encoder(torch.cat((output_10, output_56)))
        output_57 = torch.tanh(output_57)
        output_58 = self.encoder(torch.cat((output_52, output_57)))
        output_58 = torch.tanh(output_58)
        output_4 = inp[4]
        output_0 = inp[0]
        output_1 = inp[1]
        output_2 = inp[2]
        output_3 = inp[3]
        output_59 = self.encoder(torch.cat((output_2, output_3)))
        output_59 = torch.tanh(output_59)
        output_60 = self.encoder(torch.cat((output_1, output_59)))
        output_60 = torch.tanh(output_60)
        output_61 = self.encoder(torch.cat((output_0, output_60)))
        output_61 = torch.tanh(output_61)
        output_62 = self.encoder(torch.cat((output_4, output_61)))
        output_62 = torch.tanh(output_62)
        output_63 = self.encoder(torch.cat((output_58, output_62)))
        output_63 = torch.tanh(output_63)
        output_64 = self.encoder(torch.cat((output_44, output_63)))
        output_64 = torch.tanh(output_64)
        return output_64

# class RAEUnroll(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.encoder = nn.Linear(1024, 512)
    
#     def forward(self, inp):
#         output_9 = inp[9]
#         output_4 = inp[4]
#         output_7 = inp[7]
#         output_8 = inp[8]
#         output_10 = self.encoder(torch.cat((output_7, output_8)))
#         output_10 = torch.tanh(output_10)
#         output_5 = inp[5]
#         output_6 = inp[6]
#         output_11 = self.encoder(torch.cat((output_5, output_6)))
#         output_11 = torch.tanh(output_11)
#         output_12 = self.encoder(torch.cat((output_10, output_11)))
#         output_12 = torch.tanh(output_12)
#         output_13 = self.encoder(torch.cat((output_4, output_12)))
#         output_13 = torch.tanh(output_13)
#         output_14 = self.encoder(torch.cat((output_9, output_13)))
#         output_14 = torch.tanh(output_14)
#         output_2 = inp[2]
#         output_3 = inp[3]
#         output_15 = self.encoder(torch.cat((output_2, output_3)))
#         output_15 = torch.tanh(output_15)
#         output_0 = inp[0]
#         output_1 = inp[1]
#         output_16 = self.encoder(torch.cat((output_0, output_1)))
#         output_16 = torch.tanh(output_16)
#         output_17 = self.encoder(torch.cat((output_15, output_16)))
#         output_17 = torch.tanh(output_17)
#         output_18 = self.encoder(torch.cat((output_14, output_17)))
#         output_18 = torch.tanh(output_18)
#         return output_18

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(0)
    model = RAEUnroll().cuda().eval()
    
    inputs = torch.randn([n, 512], device='cuda')
    # save_tensor_bin(os.path.join(weight_prefix, f"inputs_b{args.bs}"), inputs)
    # model.export_weight()

    if args.mode == 'eval':
        with torch.no_grad():
            if args.run_pytorch:
                test_torch_eval(model, (inputs,), args.profile)
            if args.run_sys:
                # best (loop unroll + loop in c)
                to_torch_func.NNFUSION_CODEGEN_FLAGS = {'cf_level': 2}
                workflow_fix_flag(model, f"rae_unroll_bs{args.bs}", (inputs,), args.platform, args.measure)