#!/bin/python
from __operator__ import OperatorBase

# Pure antares will not need shape inference


class TopK(OperatorBase):
    def __init__(self, input_dict=None, config_infer=None):
        self.attach_directx_hlsl_kernel(input_dict)
        super().__init__(input_dict, config_infer)
    
    # Generate a HLSL Kernels
    # How about generating host call?
    def attach_directx_hlsl_kernel(self, input_dict=None):
        pass