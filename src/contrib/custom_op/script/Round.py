#!/bin/python
from __operator__ import OperatorBase, OperatorConfigSameAsInput, OperatorSingleOutputAsOneInput

# Pure antares will not need shape inference


class Round(OperatorBase):
    def __init__(self, input_dict=None, config_infer=None):
        self.attach_antares_ir(input_dict)
        super().__init__(input_dict, config_infer)

    def attach_antares_ir(self, input_json):
        input = input_json["input"]["shape"][0]
        input_size = len(input)
        lst = ", ".join([chr(c) for c in range(ord('M'), ord('M')+input_size)])
        self["antares_ir"] = "output0[{0}] = (input0[{0}] + 0.5).cast(`int32`).cast(`float16`)".format(lst)