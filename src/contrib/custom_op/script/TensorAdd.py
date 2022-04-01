import numpy as np
import sys
import random
from __operator__ import OperatorBase, OperatorTestBase

# Pure antares will not need shape inference
hlsl_kernel_add = '''
RWStructuredBuffer<float> input0: register(u0);
RWStructuredBuffer<float> input1: register(u1);
RWStructuredBuffer<float> output0: register(u2);

[numthreads(__threads__, 1, 1)] void CSMain(uint3 gid: SV_GroupID, uint3 tid: SV_GroupThreadID, uint gidx: SV_GroupIndex) {
    // SV_GroupID in range of dipatch(GroupID)
    // SV_GroupThreadID is the place in numthreads(x,x,x)
    // SV_GroupIndex = SV_GroupThreadID.z*dimx*dimy + SV_GroupThreadID.y*dimx + SV_GroupThreadID.x

    uint idx = gid.x * __threads__ + gidx;
    if(idx < __total_size__)
        output0[idx] = input0[idx] + input1[idx];
}
'''

class TensorAdd(OperatorBase):
    def __init__(self, input_dict=None, config_infer=None):
        # Here is some difference with original ONNX define
        super().__init__(input_dict, self.config_infer)
        self.attach_directx_hlsl_kernel(input_dict)

    def attach_directx_hlsl_kernel(self, input_dict=None):
        shape = input_dict["input"]["shape"][0]
        size = 1
        for e in shape:
            size = size * e 
        threads = 1024
        import math
        blocks = math.ceil(size / threads)
        self["hlsl_kernel"] = hlsl_kernel_add.replace("__threads__", str(threads)).replace("__total_size__", str(size))
        self["launch_config"] = [[blocks, 1, 1], [1024, 1, 1]]
        self["entry_piont"] = "CSMain"


    def config_infer(self, input_dict=None):
        outputs = {"shape": [], "dtype": []}
        outputs["shape"].append(input_dict["input"]["shape"][0])
        outputs["dtype"].append(input_dict["input"]["dtype"][0])
        return outputs


class TensorAddTest(OperatorTestBase, TensorAdd):
    def __init__(self, input_dict=None, config_infer=None):
        pass

    def create_add_test(self):
        shape = []
        for i in range(0, random.randint(1, 2)):
            shape.append(random.randint(1000, 2048))
        self["input"] = {}
        self["input"]["shape"] = [shape, shape]
        self["input"]["dtype"] = [["float32"]]

        a = np.ones(shape = shape, dtype = np.single)
        b = np.ones(shape = shape, dtype = np.single)
        c = a + b

        return {"kernel": TensorAdd(self), "input": [a, b], "output": [c]}
    
    def create_add_test_copy(self):
        shape = []
        for i in range(0, random.randint(1, 2)):
            shape.append(random.randint(1000, 2048))
        self["input"] = {}
        self["input"]["shape"] = [shape, shape]
        self["input"]["dtype"] = [["float32"]]

        a = np.ones(shape = shape, dtype = np.single)
        b = np.ones(shape = shape, dtype = np.single)
        c = a + b

        return {"kernel": TensorAdd(self), "input": [a, b], "output": [c]}