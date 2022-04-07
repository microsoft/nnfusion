#!/bin/python
from __operator__ import OperatorBase, OperatorConfigSameAsInput, OperatorSingleOutputAsOneInput

# $ python script --operator-name="CustomOP" --input-config="{\"operator\":\"CustomOP\",\"input\":{\"shape\":[[1,2],[2,3]], \"dtype\":[1,2]}}"
# {"operator": "CustomOP", "input": {"shape": [[1, 2], [2, 3]], "dtype": [1, 2]}, "write_stdout": true, "output": {"shape": [[1, 2], [2, 3]], "dtype": [1, 2]}}


class CustomOP(OperatorConfigSameAsInput):
    def __init__(self, input_dict=None, config_infer=None):
        super().__init__(input_dict, config_infer)

# $ python script --operator-name="CustomOP1" --input-config="{\"operator\":\"CustomOP\",\"input\":{\"shape\":[[1,2],[2,3]], \"dtype\":[1,2]}}"
# {"operator": "CustomOP", "input": {"shape": [[1, 2], [2, 3]], "dtype": [1, 2]}, "output": {"shape": [], "dtype": []}}


class CustomOP1(OperatorConfigSameAsInput):
    def __init__(self, input_dict=None, config_infer=None):
        super().__init__(input_dict, self.config_infer)

    def config_infer(self, input_json):
        return {"shape": [], "dtype": []}

# $ python script --operator-name="CustomOP2" --input-config="{\"operator\":\"CustomOP\",\"input\":{\"shape\":[[1,2],[2,3]], \"dtype\":[1,2]}}"
# {"operator": "CustomOP", "input": {"shape": [[1, 2], [2, 3]], "dtype": [1, 2]}, "write_stdout": true, "output": {"shape": [[2, 3]], "dtype": [2]}}


class CustomOP2(OperatorSingleOutputAsOneInput):
    def __init__(self, input_dict=None, config_infer=None):
        super().__init__(input_dict, config_infer, 1)


class CustomOP3(OperatorBase):
    def __init__(self, input_dict=None, config_infer=None):
        super().__init__(input_dict, self.config_infer)

    def config_infer(self, input_json):
        self['antares_ir'] = "output0[M, N] = input0[M, N] * 0.001"
        self["cuda_kernel"] = "{ int index = blockIdx.x*blockDim.x + threadIdx.x; output0[index] = input0[index] * @custom_value@; }"
        self["launch_config"] = "[[1, 1, 4], [1, 1, 32]]"
        return {"shape": [], "dtype": []}