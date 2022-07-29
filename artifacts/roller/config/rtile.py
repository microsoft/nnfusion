import tvm
from tvm import te
from roller.utils import *
import inspect
from copy import deepcopy

def build_name2val(op, shape):
    axis = op.axis
    reduce_axis = op.reduce_axis

    ret = {}
    for i in range(len(axis)):
        ret[axis[i].var.name] = shape[i]
    for i in range(len(reduce_axis)):
        ret[reduce_axis[i].var.name] = shape[i + len(axis)]

    return ret

def extract_producer_load(expr):
    if isinstance(expr, tvm.tir.ProducerLoad):
        return [expr]
    elif isinstance(expr, tvm.tir.Mul):
        return extract_producer_load(expr.a) + extract_producer_load(expr.b)
    elif isinstance(expr, tvm.tir.LT):
        return extract_producer_load(expr.a) + extract_producer_load(expr.b)
    elif isinstance(expr, tvm.tir.FloatImm):
        return []
    else:
        print(type(expr))
        assert(False)

def eval_index_len(index_expr, name2val):
    if isinstance(index_expr, tvm.tir.Var):
        return name2val[index_expr.name]
    elif isinstance(index_expr, tvm.tir.Add):
        return eval_index_len(index_expr.a, name2val) + eval_index_len(index_expr.b, name2val) - 1
    else:
        print(type(index_expr))
        assert(False)

def build_tensors(op, shape):
    name2val = build_name2val(op, shape)

    assert(len(op.body) == 1)
    compute_expr = op.body[0]

    def calc_tensor_dim(load_expr):
        indices = load_expr.indices
        dims = [0] * len(indices)
        for i in range(len(indices)):
            dims[i] = eval_index_len(indices[i], name2val)
        return dims

    if isinstance(compute_expr, tvm.tir.Reduce):
        in_tensors = {}
        for in_expr in compute_expr.source:
            load_exprs = extract_producer_load(in_expr)
            for load_expr in load_exprs:
                producer_name = load_expr.producer.name
                dims = calc_tensor_dim(load_expr)
                assert(producer_name not in in_tensors)
                in_tensors[producer_name] = dims

        in_tensors = list(in_tensors.items())
    elif isinstance(compute_expr, tvm.tir.Call):
        load_exprs = []
        for arg in compute_expr.args:
            load_exprs = load_exprs + extract_producer_load(arg)
        in_tensors = {}
        for load_expr in load_exprs:
            producer_name = load_expr.producer.name
            dims = calc_tensor_dim(load_expr)
            if producer_name in in_tensors:
                assert(dims == in_tensors[producer_name])
            else:
                in_tensors[producer_name] = dims
        in_tensors = list(in_tensors.items())
    else:
        assert(False)

    out_tensors = [(op.output(0).name, shape[:len(op.axis)])]

    return in_tensors, out_tensors


class rTile:
    def __init__(self, expr, shape, saxis, raxis, tvm_out_tensor):
        self.expr = expr
        self.shape = shape
        # todo: what if multi-output tensors?
        self.tvm_out_tensor = tvm_out_tensor
        self.ori_in = []
        self.pad_in = []
        self.outs = []
        self.unpad_outs = []
        # self.expr_out = self.expr(self.shape, for_rtile=True)
        # self.input_tensors = self.expr_out[0]
        # self.output_tensors =self.expr_out[1]
        self.input_tensors, self.output_tensors = build_tensors(expr, shape)

        for it in self.input_tensors:
            if '_pad' in it[0]:
                self.pad_in.append(it)
            else:
                self.ori_in.append(it)

        for ot in self.output_tensors:
            if '_unpad' in ot[0]:
                self.unpad_outs.append(ot)
            else:
                self.outs.append(ot)
        # self.saxis, self.raxis = get_axis_names(self.tvm_out_tensor)
        self.saxis = saxis
        self.raxis = raxis
        self.spatial_dim = len(self.saxis)

        self.axis_shape_map = {}
        aid = 0
        for axis_name in self.saxis:
            self.axis_shape_map[axis_name] = self.shape[aid]
            aid += 1
        for axis_name in self.raxis:
            self.axis_shape_map[axis_name] = self.shape[aid]
            aid += 1

        if len(self.pad_in) > 0:
            self.storage_padding = [[0 for _ in range(len(t[1]))] for t in self.pad_in]
        else:
            self.storage_padding = [[0 for _ in range(len(t[1]))] for t in self.input_tensors]

        # get input axis name
        all_axis_name = set(self.saxis + self.raxis)
        string = str(self.tvm_out_tensor.op)
        source_match = 'source=\[.+\], init=\['
        x = re.search(source_match, string)
        if not x:
            body_match = 'body=\[.+\], init=\['
            x = re.search(body_match, string)
            if not x:
                body_match = 'body=\[.+\], axis=\['
                x = re.search(body_match, string)
                assert x
                source = x.group()[6:-9]
        else:
            source = x.group()[8: -9]
        ins_axis_string = []

        find = False
        count = 0
        left = -1
        # for i in range(len(source)):
        #     if find and count == 0:
        #         ins_axis_string.append(source[left: i])
        #         find = False
        #         count = 0
        #         left = -1
        #     else:
        #         if source[i] == '[':
        #             count += 1
        #             find = True
        #             if left == -1:
        #                 left = i
        #         elif source[i] == ']':
        #             count -= 1
        #             if find and count == 0:
        #                 ins_axis_string.append(source[left: i + 1])
        #                 left = -1
        #                 find = False

        for i in range(len(source)):
            if not find or count != 0:
                if source[i] == '[':
                    count += 1
                    find = True
                    if left == -1:
                        left = i
                elif source[i] == ']':
                    count -= 1
                    if find and count == 0:
                        ins_axis_string.append(source[left: i + 1])
                        left = -1
                        find = False

        self.ins_axis_name = []
        for i in range(len(ins_axis_string)):
            ias = ins_axis_string[i][1:-1]
            names = []
            ias = ias.split(', ')
            for a in ias:
                a = re.split(r'([-+*/()%])|\s+', a)
                for n in a:
                    if n in all_axis_name:
                        names.append(n)
            self.ins_axis_name.append(names)

    def GetInputDataTiles(self):
        ret = []
        for tensor in self.GetInputTensors():
            ret.append(tensor[1])
        return ret

    def GetOutputDataTiles(self):
        # todo: what if multi-outputs?
        ret = []
        for tensor in self.GetOutputTensors():
            ret.append(tensor[1])
        return ret

    def Dump(self):
        spatial_shape = self.shape[:len(self.saxis)]
        reduce_shape = self.shape[len(self.saxis):]
        return "[tile: {}; step: {}]".format(spatial_shape, reduce_shape)

    def Dimensions(self):
        return self.shape

    def SAxis(self):
        return self.saxis

    def RAxis(self):
        return self.raxis

    def InputAxis(self):
        return self.ins_axis_name

    def SDimensions(self):
        return self.shape[:self.spatial_dim]

    def RDimensions(self):
        return self.shape[self.spatial_dim:]

    def Size(self):
        ret = 1
        for d in self.SDimensions():
            ret *= d
        return ret

    def UpdateStoragePadding(self, align_info):
        inputs_shape = self.GetInputDataTiles()
        for info in align_info:
            idx, factor, offset = info
            if len(inputs_shape[idx]) >= 2:
                stride = inputs_shape[idx][-1]
                padding = (factor + offset - stride % factor) % factor
                self.storage_padding[idx][-1] = padding
        return

    def GetStoragePadding(self):
        return self.storage_padding

    def copy(self):
        newTile = rTile(self.expr, self.shape, self.saxis, self.raxis, self.tvm_out_tensor)
        newTile.storage_padding = self.storage_padding.copy()
        return newTile

    def GetInputTensors(self):
        return self.pad_in if len(self.pad_in) > 0 else self.input_tensors

    def GetOutputTensors(self):
        return self.unpad_outs if self.unpad_outs else self.output_tensors

    def GetAxisLen(self, axis_name):
        assert axis_name in self.axis_shape_map
        return self.axis_shape_map[axis_name]
