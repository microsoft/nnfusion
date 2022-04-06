import numpy as np
import sys
import random
from __operator__ import OperatorBase, OperatorTestBase, get_type_info, get_antares_type_str

# Pure antares will not need shape inference
hlsl_kernel_add = '''
// Input should be organized from t0 as StructuredBuffer, and output should be RWStructuredBuffer from u0
StructuredBuffer<float> input0: register(t0);
RWStructuredBuffer<float> input1: register(u0);
RWStructuredBuffer<float> output0: register(u1);

// numthreads will be replaced by the comment below in nnfusion
[numthreads(1, 1, 1)] void CSMain(uint3 gid: SV_GroupID, uint3 tid: SV_GroupThreadID, uint gidx: SV_GroupIndex) {
    // [thread_extent] blockIdx.x =  __blocks__
    // [thread_extent] threadIdx.x = __threads__
    uint idx = gid.x * __threads__ + gidx;
    if(idx < __total_size__)
        output0[idx] = input0[idx] + input1[idx];
}
'''

'''
# Input JSON:
{
    "input":{
      "shape":[[1755],[1755]],
      "dtype":["float32", "float32"]
    },
    "output":{
      "shape":[[1755]],
      "dtype":["float32"]
    },
    "data" : {}
}

# There could be "data" : { 1: [1., 2., 3.]} to indicate which input is constant and the value will be organized as list.

# Output JSON:
{
    "input":{
      "shape":[[1755],[1755]],
      "dtype":["float32", "float32"]
    },
    "output":{
      "shape":[[1755]],
      "dtype":["float32"]
    },
    "hlsl_kernel":"// GLOBALS: input0:float32[1755], input1:float32[1755] -> output0:float32[1755]\n// LOCAL: CSMain -- input0:float32[1755], input1:float32[1755] -> output0:float32[1755]\n\nStructuredBuffer<float> input0: register(t0);\nRWStructuredBuffer<float> input1: register(u0);\nRWStructuredBuffer<float> output0: register(u1);\n\n[numthreads(1, 1, 1)] void CSMain(uint3 gid: SV_GroupID, uint3 tid: SV_GroupThreadID, uint gidx: SV_GroupIndex) {\n    // [thread_extent] blockIdx.x =  __blocks__\n    // [thread_extent] threadIdx.x = 1024\n    uint idx = gid.x * 1024 + gidx;\n    if(idx < 1755)\n        output0[idx] = input0[idx] + input1[idx];\n}\n",
    "launch_config":[[2,1,1], [1024,1,1]],
    "entry_point":"CSMain"
}
'''


class TemplateHLSLOperator(OperatorBase):
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
        self["hlsl_kernel"] = hlsl_kernel_add.replace(
            "__threads__", str(threads)).replace("__total_size__", str(size))
        self["launch_config"] = [[blocks, 1, 1], [1024, 1, 1]]
        self["entry_point"] = "CSMain"

        # Add antares_info to be parsed in codegen
        antares_info = "// GLOBALS: input0:{0}[{1}], input1:{2}[{3}] -> output0:{4}[{5}]".format(
            get_antares_type_str(self["input"]["dtype"][0]),
            ", ".join([str(i) for i in self["input"]["shape"][0]]),
            get_antares_type_str(self["input"]["dtype"][1]),
            ", ".join([str(i) for i in self["input"]["shape"][1]]),
            get_antares_type_str(self["output"]["dtype"][0]),
            ", ".join([str(i) for i in self["output"]["shape"][0]]),
        )

        self["hlsl_kernel"] = antares_info + "\n" + \
            antares_info.replace(
                "GLOBALS:", "LOCAL: CSMain --") + "\n" + self["hlsl_kernel"]

    def config_infer(self, input_dict=None):
        outputs = {"shape": [], "dtype": []}
        outputs["shape"].append(input_dict["input"]["shape"][0])
        outputs["dtype"].append(input_dict["input"]["dtype"][0])
        return outputs


class TemplateHLSLOperatorTest(OperatorTestBase, TemplateHLSLOperator):
    def __init__(self, input_dict=None, config_infer=None):
        pass

    def create_add_test(self):
        shape = []
        for i in range(0, random.randint(1, 2)):
            shape.append(random.randint(1000, 2048))
        self["input"] = {}
        self["input"]["shape"] = [shape, shape]
        self["input"]["dtype"] = ["float32", "float32"]

        a = np.ones(shape=shape, dtype=np.single)
        b = np.ones(shape=shape, dtype=np.single)
        c = a + b

        return {"kernel": TemplateHLSLOperator(self), "input": [a, b], "output": [c]}

    def create_add_test_copy(self):
        shape = []
        for i in range(0, random.randint(1, 2)):
            shape.append(random.randint(1000, 2048))
        self["input"] = {}
        self["input"]["shape"] = [shape, shape]
        self["input"]["dtype"] = ["float32", "float32"]

        a = np.ones(shape=shape, dtype=np.single)
        b = np.ones(shape=shape, dtype=np.single)
        c = a + b

        return {"kernel": TemplateHLSLOperator(self), "input": [a, b], "output": [c]}
