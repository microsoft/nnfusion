import os
import json
import re
import argparse
import importlib.machinery
from types import FunctionType, MethodType
import numpy as np
import logging

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.WARN
)

# This is to provide Operator interface
# It will take input and output kernel code

op_map = None

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

class OperatorTestBase(dict):
    def __init__(self):
        self.name = "Default"
        self.test_cases = []
    
    def allclose(self, truth, output):
        no_error_flag = True
        for i in range(0, len(truth)):
            if not np.allclose(truth[i], output[i]):
                logging.error("numpy.allclose failed at output {0}. {1} vs {2}".format(i, truth[i], output[i]))
                no_error_flag = False
        return no_error_flag

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
def load_operators(baseclasses = OperatorBase):
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
                    if type(baseclasses) is list:
                        isasubclass = True
                        for baseclass in baseclasses:
                            if isinstance(element, type) or not issubclass(element, baseclass) and element is baseclass:
                                isasubclass = False
                            print(element, baseclass, issubclass(element, baseclass), element is baseclass)
                        if isasubclass:
                            operator_map[element.__name__] = element
                    else:
                        baseclass = baseclasses
                        if isinstance(element, type) and issubclass(element, baseclass) and element is not baseclass:
                            operator_map[element.__name__] = element
            except:
                pass
    return operator_map

def load_operator_by_name(op_name):
    global op_map
    if op_map is None:
        op_map = load_operators()
    if op_name in op_map.keys():
        op = op_map[op_name]
        return op
    else:
        return None

def get_operator_config(op_name, conf_dict):
    op_type = load_operator_by_name(op_name)
    if op_type is not None:
        return op_type(conf_dict)
    return {}

def get_operator_tests(op_name):
    global op_map
    op_type = load_operator_by_name(op_name)
    test_cases = []
    for op in op_map:
        if issubclass(op_map[op], op_type) and issubclass(op_map[op], OperatorTestBase):
            # Generate all tests
            op_obj = op_map[op]()
            F = [getattr(op_obj, m) for m in dir(op_obj) if not m.startswith('__')]
            allclose = None
            for ac in F:
                if ac.__name__ == "allclose":
                   allclose = ac
                   break
            for create_test_method in [f for f in F if type(f) is MethodType and f.__name__.startswith("create_")]:
                test_case = create_test_method()
                test_case["test"] = create_test_method.__name__[7:]
                if "allclose" not in test_case:
                    test_case["allclose"] = allclose
                test_cases.append(test_case)
            
            
    return test_cases

def get_type_info(typestr):
    if typestr == "half" or typestr == "float16":
        return ("float16_t", 2, np.finfo(np.float16).min, np.finfo(np.float16).max)
    if typestr == "float32" or typestr == "float":
        return ("float", 4, np.finfo(np.float32).min, np.finfo(np.float32).max)
    if typestr == "double":
        return ("double", 8, np.finfo(np.double).min, np.finfo(np.double).max)
    if typestr == "int" or typestr == "int32" or typestr == "int32_t":
        return ("int", 4, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    if typestr == "int64" or typestr == "long long" or typestr == "int64_t":
        return ("int64_t", 8, np.iinfo(np.int64).min, np.iinfo(np.int64).max)
    exit(-1)

def get_antares_type_str(typestr):
    if typestr == "half" or typestr == "float16":
        return "float16"
    if typestr == "float32" or typestr == "float":
        return "float32"
    if typestr == "double":
        return "float64"
    if typestr == "int" or typestr == "int32":
        return "int32"
    if typestr == "int64" or typestr == "int64_t" or typestr == "long long":
        return "int64"

def read_file(file_name):
    with open(os.path.join(os.path.dirname(__file__), file_name)) as f:
        lines = f.readlines()
        return "".join(lines)

def list_template_args(content_str):
    # match and list all variables begin with "__" and end with "__"
    res = re.findall(r"__\w+__", content_str)
    return set(res)

def replace_template_args(content_str, parameters):
    tkeys = list_template_args(content_str)
    for tkey in list(tkeys):
        content_str = content_str.replace(tkey, parameters[tkey])
    return content_str