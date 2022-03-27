import os
import json
import argparse
import importlib.machinery

# This is to provide Operator interface
# It will take input and output kernel code


class OperatorBase(dict):
    def __init__(self, input_dict=None, config_infer=None):
        # The field an operator should have
        for k in input_dict:
            self[k] = input_dict[k]

        if config_infer is None:
            self["output"] = self.config_infer(self)
        else:
            self["output"] = config_infer(self)

    def config_infer(self, input_json):
        outputs = {"shape": [], "dtype": []}
        for ele in input_json["input"]["shape"]:
            outputs["shape"].append(ele)
        for ele in input_json["input"]["dtype"]:
            outputs["dtype"].append(ele)
        return outputs


class OperatorConfigSameAsInput(OperatorBase):
    def __init__(self, input_dict=None, config_infer=None):
        super().__init__(input_dict, config_infer)


class OperatorSingleOutputAsOneInput(OperatorBase):
    def __init__(self, input_dict=None, config_infer=None, as_pos=0):
        self.as_pos = as_pos
        super().__init__(input_dict, self.config_infer)

    def set_as_pos(self, as_pos):
        self.as_pos = as_pos

    def config_infer(self, input_json):
        outputs = {"shape": [], "dtype": []}
        ele = input_json["input"]["shape"][self.as_pos]
        outputs["shape"].append(ele)
        ele = input_json["input"]["dtype"][self.as_pos]
        outputs["dtype"].append(ele)
        return outputs


# Try loading every operators
def load_operators():
    operator_map = dict()
    for file in os.listdir(os.path.dirname(__file__)):
        if file.endswith(".py") and not file.startswith("__"):
            try:
                hname = file[:-3]
                loader = importlib.machinery.SourceFileLoader(
                    hname, os.path.join(os.path.dirname(__file__), file))
                handle = loader.load_module(hname)
                for element in dir(handle):
                    element = getattr(handle, element)
                    if isinstance(element, type) and issubclass(element, OperatorBase) and element is not OperatorBase:
                        operator_map[element.__name__] = element
            except:
                pass
    return operator_map


def get_operator_config(op_name, conf_dict):
    op_map = load_operators()
    if op_name in op_map.keys():
        op = op_map[op_name](conf_dict)
        return op
    return {}
