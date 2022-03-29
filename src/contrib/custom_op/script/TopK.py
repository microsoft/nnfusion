#!/bin/python
import numpy as np
import sys
from __operator__ import OperatorBase, OperatorTestBase

# Pure antares will not need shape inference

hlsl_kernel = '''
// @dtype@ : input/output datatype
// @BITONIC_BLOCK_SIZE@ : How many blocks
RWStructuredBuffer<@dtype@> Input : register( t0 );
RWStructuredBuffer<@dtype@> Data : register( u0 );

groupshared @dtype@ shared_data[@BITONIC_BLOCK_SIZE@];

[numthreads(@BITONIC_BLOCK_SIZE@, 1, 1)]
void BitonicSort( uint3 Gid : SV_GroupID, 
                  uint3 DTid : SV_DispatchThreadID, 
                  uint3 GTid : SV_GroupThreadID, 
                  uint GI : SV_GroupIndex )
{
    // Load shared data
    shared_data[GI] = Data[DTid.x];
    GroupMemoryBarrierWithGroupSync();

    // Sort the shared data
    for (unsigned int j = g_iLevel >> 1 ; j > 0 ; j >>= 1)
    {
        @dtype@ result = ((shared_data[GI & ~j] <= shared_data[GI | j]) == (bool)(g_iLevelMask & DTid.x))? shared_data[GI ^ j] : shared_data[GI];
        GroupMemoryBarrierWithGroupSync();
        shared_data[GI] = result;
        GroupMemoryBarrierWithGroupSync();
    }
    
    // Store shared data
    Data[DTid.x] = shared_data[GI];
}
'''

# ONNX Reference Link:https://github.com/onnx/onnx/blob/main/docs/Operators.md#topk


class TopK(OperatorBase):
    def __init__(self, input_dict=None, config_infer=None):
        # Here is some difference with original ONNX define
        super().__init__(input_dict, self.config_infer)
        self.attach_directx_hlsl_kernel(input_dict)

    # Generate a HLSL Kernels
    # How about generating host call?
    def attach_directx_hlsl_kernel(self, input_dict=None):
        pass

    def config_infer(self, input_dict=None):
        if len(input_dict["input"]["shape"]) == 1:
            sys.stderr.write("TopK only support one input: K should be translated to constant attribution.")
            exit(-1)
        outputs = {"shape": [], "dtype": []}
        for ele in input_dict["input"]["shape"]:
            outputs["shape"].append(ele)
        for ele in input_dict["input"]["dtype"]:
            outputs["dtype"].append(ele)
        
        if self['axis'] < 0:
                self['axis'] += len(outputs["shape"][0])
        
        outputs["shape"][0][self['axis']] = input_dict['K']
        return outputs


class TopKTest(OperatorTestBase):
    def __init__(self, input_dict=None, config_infer=None):
        # <Default Value>
        # Attribute
        self["axis"] = -1 # Dimension on which to do the sort. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).
        self["largest"] = 1 # Whether to return the top-K largest or smallest elements.
        self["sorted"] = 1  # Whether to return the elements in sorted order.

        # <Inputs>
        # self["x"]  # Tensor of shape [a_1, a_2, ..., a_n, r]
        # self["K"]  # A 1-D tensor containing a single positive value corresponding to the number of top elements to retrieve

        # <Outputs>
        # Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] containing top K values from the input tensor
        # self["Values"]
        # Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] containing the corresponding input tensor indices for the top K values.
        # self["Indices"]

        self.test_cases.append(self.create_topk)
        self.name = "TopK"

    def create_topk(self):
        self["axis"] = 1
        self["largest"] = 1
        self.X = np.array([[0, 1, 2, 3],[4, 5, 6, 7],[8, 9, 10, 11],], dtype=np.float32)
        self.K = 3
        self.values_ref = np.array([[3,  2,  1], [7,  6,  5], [11, 10,  9]], dtype=np.float32)
        self.indicies_ref = np.array([[3,  2,  1], [3, 2, 1], [3, 2, 1]], dtype=np.int64)
        self["input"]["shape"] = [[3, 4], [1]]
        self["input"]["dtype"] = [["float32"], ["int64"]]

        return (TopK(self), self)