import tvm
import lang
import numpy as np
from tvm import arith

class Edge:
    def __init__(self, src_node, dst_node, src_id, dst_id):
        self._src_node = src_node
        self._dst_node = dst_node
        self._src_id = src_id
        self._dst_id = dst_id

    @property
    def src_node(self):
        return self._src_node

    @property
    def dst_node(self):
        return self._dst_node

    @property
    def src_id(self):
        return self._src_id

    @property
    def dst_id(self):
        return self._dst_id

class Node:
    def __init__(self, inputs, name):
        self.name = name
        self._out_edges = []
        self._in_edges = []
        self._shapes = []
        self._dtypes = []
        self._tag = {}

        for i, node in enumerate(inputs):
            if node is None:
                inputs[i] = PlaceHolderNode("input" + str(i))

        for dst_id, n in enumerate(inputs):
            if isinstance(n, Node):
                n = (n, 0)
            assert(len(n) == 2)
            src_node, src_id = n[0], n[1]
            edge = Edge(src_node, self, src_id, dst_id)
            self._in_edges.append(edge)
            src_node._out_edges.append(edge)

    def emit_config(self):
        raise NotImplementedError

    @property
    def inputs(self):
        return self._in_edges

    @property
    def outputs(self):
        return self._out_edges

    def set_inputs(self, i, edge):
        assert i < len(self._in_edges)
        self._in_edges[i] = edge

    def set_outputs(self, i, edge):
        assert i < len(self._out_edges)
        self._out_edges[i] = edge

    def get_shape(self, id=0):
        return self._shapes[id]

    def set_shape(self, shape, id=0, overwrite=False):
        if len(self._shapes) <= id:
            self._shapes.extend([None for _ in range(id - len(self._shapes) + 1)])
        elif self._shapes[id] is not None and not overwrite:
            assert self._shapes[id] == list(map(int, shape)), (self._shapes, list(map(int, shape)))
        self._shapes[id] = list(map(int, shape))

    def get_dtype(self, id=0):
        return self._dtypes[id]

    def set_dtype(self, dtype: tvm.DataType, id=0):
        assert isinstance(dtype, tvm.DataType), type(dtype)
        if len(self._dtypes) <= id:
            self._dtypes.extend([None for _ in range(id - len(self._dtypes) + 1)])
        elif self._dtypes[id] is not None:
            assert self._dtypes[id] == dtype, (self._dtypes, dtype)
        self._dtypes[id] = dtype

    def is_placeholder(self):
        return False

    def is_output(self):
        return False

    def add_tag(self, k, v=True):
        self._tag[k] = v

    def get_tag(self, k):
        if k not in self._tag:
            return None
        return self._tag[k]

    def num_outputs(self):
        if len(self.outputs) == 0:
            return 0
        return max([e.src_id for e in self.outputs]) + 1

    def __repr__(self) -> str:
        return "<Node, " + self.name + ">"

class PlaceHolderNode(Node):
    def __init__(self, name):
        super().__init__([], "PlaceHolder " + name)

    def is_placeholder(self):
        return True

class OutputNode(Node):
    def __init__(self, node, id=0):
        super().__init__([(node, id)], "Output ")
        self.set_shape(node.get_shape(id))
        self.set_dtype(node.get_dtype(id))

    def infer_dependency(self, shape, rstep={}):
        return {0 : shape}

    def is_output(self):
        return True

class MatMulNode(Node):
    def __init__(self, inputs, n, m ,k):
        super().__init__(inputs, "MatMul")
        from op import MatmulOp
        from .tvm_ops import tvm_matmul
        self.op = MatmulOp(m, k, n)
        self.args = tvm_matmul(n, m, k)

class ConvNode(Node):
    def __init__(self, inputs, n, c, h, w, f, k, s=1, d=1, p="SAME"):
        super().__init__(inputs, "Conv")
        from op import ConvOp
        from .tvm_ops import tvm_conv
        self.op = ConvOp(n, c, f, k, s, h, w, d, p)
        self.args = tvm_conv(n, c, h, w, f, k, s, d, p)

class DepthwiseConvNode(Node):
    def __init__(self, inputs, n, c, h, w, k, s=1, d=1, p="SAME", m=1):
        super().__init__(inputs, "Conv")
        from op import DepthwiseConvOp
        from .tvm_ops import tvm_depthwise_conv
        self.op = DepthwiseConvOp(n, c, k, s, h, w, d, p, m)
        self.args = tvm_depthwise_conv(n, c, h, w, k, s, d, p, m)

class IRNode(Node):
    def __init__(self, inputs, antares_ir, name="Compute"):
        super().__init__(inputs, name)
        self.ir = antares_ir
        self._input_args, self._output_args = lang.translate_ir_to_tvm(self.ir)
        self._output_names = [arg.name for arg in self._output_args]
        if len(self._output_args) > 1:
            self._process_multiple_output()
        if len(self._input_args) < len(self.inputs):
            # some placeholders are extra info that might not be used in tensor computation
            new_input_edges = []
            for arg in self._input_args:
                name = arg.name
                assert name.startswith('input')
                input_edge = self.inputs[int(name[5:])]
                new_input_edges.append(input_edge)
            self._in_edges = new_input_edges
        self.args = self._input_args + self._output_args
        self.ana = lang.get_analyzer_by_ir(self.ir)
        for edge, arg in zip(self.inputs, self.args):
            edge.src_node.set_shape(arg.shape, edge.src_id)
            edge.src_node.set_dtype(tvm.DataType(arg.dtype), edge.src_id)
        for output_id, arg in enumerate(self._output_args):
            self.set_shape(arg.shape, output_id)
            self.set_dtype(tvm.DataType(arg.dtype), output_id)
        self._extract_axis()
        self._sche = self.create_schedule()

    def infer_dependency(self, shape, rstep={}):
        shape = {name: [tvm.arith.ConstIntBound(0, val - 1) for val in shape] for name in self._output_names}
        shapes = self.ana.infer(shape, rstep)
        result = {}
        for i in range(len(self.inputs)):
            name = self._input_args[i].name
            shape = shapes[name]
            # should not exceed original shape
            result[i] = list(map(min, zip(shape, self.inputs[i].src_node.get_shape())))
        return result

    def infer_smem_usage(self, shape, rstep):
        result = 0
        shape = {name: [tvm.arith.ConstIntBound(0, val - 1) for val in shape] for name in self._output_names}
        shapes = self.ana.infer(shape, rstep)
        cached_tensor = set()
        for op in self._sche.stage_map:
            if not isinstance(op, tvm.te.ComputeOp):continue
            for tensor in op.input_tensors:
                cache = isinstance(self._sche[tensor].op, tvm.te.PlaceholderOp) \
                    and len(op.output(0).shape) > len(tensor.shape) \
                    and np.prod(op.output(0).shape) > np.prod(tensor.shape) # is broadcast
                if len(op.reduce_axis) > 0:
                    cache = True
                if cache:
                    cached_tensor.add(tensor)
        for tensor in cached_tensor:
            if tensor.name.startswith("input"):
                input_id = [arg.name for arg in self._input_args].index(tensor.name)
                assert(input_id >= 0)
                src_node = self.inputs[input_id].src_node
                if not src_node.is_placeholder():
                    continue
            result += np.prod(shapes[tensor.name]) * int(tvm.DataType(tensor.dtype).bits // 8)
        return result

    def block_infer(self, tile_map, block_expr, block_idx):
        space_expr = []
        grid_size = 1
        for ax_len, tile_len in zip(reversed(self.get_shape()), reversed(tile_map[self])):
            num_block = int(np.ceil(ax_len / tile_len))
            grid_size *= num_block
            space_expr.append(block_expr % num_block * tile_len)
            block_expr = block_expr // num_block
        output_exprs = {name : reversed(space_expr) for name in self._output_names}
        input_exprs = self.ana.get_input_exprs(output_exprs)
        result = {}
        ana = arith.Analyzer()
        ana.update(block_idx, arith.ConstIntBound(0, grid_size - 1))
        for i in range(len(self.inputs)):
            block_expr = 0
            inode = self.inputs[i].src_node
            if isinstance(inode, PlaceHolderNode):
                continue
            for expr, ax_len, tile_len in zip(input_exprs[self._input_args[i].name], inode.get_shape(), tile_map[inode]):
                num_block = int(np.ceil(ax_len / tile_len))
                block_expr = block_expr * num_block + tvm.te.max(expr // tile_len, 0)
            result[i] = ana.simplify(block_expr)
        return result

    # axis name -> axis length
    def _extract_axis(self):
        queue = self._output_args.copy()
        self.raxis = {}
        visited = {item : True for item in queue}
        while len(queue) > 0:
            t = queue.pop(0)
            if isinstance(t.op, tvm.te.PlaceholderOp):
                continue
            for axis in t.op.reduce_axis:
                assert(str(axis.var.name) not in self.raxis), axis.var.name
                self.raxis[str(axis.var.name)] = int(axis.dom.extent)
            for it in t.op.input_tensors:
                if not it in visited:
                    visited[it] = True
                    queue.append(it)

        self.saxis = {}
        for axis in self._output_args[0].op.axis:
            assert(str(axis.var.name) not in self.saxis), axis.var.name
            assert(str(axis.var.name) not in self.raxis), axis.var.name
            self.saxis[str(axis.var.name)] = int(axis.dom.extent)

    def create_schedule(self):
        args = self._output_args
        return tvm.te.create_schedule([x.op for x in args])

    def _process_multiple_output(self):
        layout = ", ".join([ax.var.name for ax in self._output_args[0].op.axis])
        sandbox = {"args" : self._output_args}
        exec("func=lambda {}: [op[{}] for op in args]".format(layout, layout), sandbox)
        args = tvm.te.compute(self._output_args[0].shape, sandbox["func"], name="output_proxy")
        self._output_args = list(args)
        self.args = self._input_args + self._output_args

    def clone(self, inputs):
        new_node = IRNode(inputs, self.ir, self.name)
        for k, v in self._tag.items():
            new_node.add_tag(k, v)
        return new_node


def topo_order(list_of_nodes):
    input_ready_count = {node : len(node.inputs) for node in list_of_nodes}
    ready = list(filter(lambda node : input_ready_count[node] == 0, list_of_nodes))
    output_list = []
    while len(ready) > 0:
        node = ready.pop(0)
        output_list.append(node)
        for edge in node.outputs:
            dst_node = edge.dst_node
            if dst_node not in input_ready_count:
                input_ready_count[dst_node] = len(dst_node.inputs)
                list_of_nodes.append(dst_node)
            input_ready_count[dst_node] -= 1
            assert(input_ready_count[dst_node] >= 0)
            if input_ready_count[dst_node] == 0:
                ready.append(dst_node)
    assert(len(list_of_nodes) == len(output_list))
    return output_list

def find_topo_sort_priority(output_node_list):
    import sys
    sys.setrecursionlimit(10000)
    def topo_sort_get_layer(node, topo_layer):
        if node in topo_layer:
            return
        topo_layer[node] = 0
        for edge in node.inputs:
            topo_sort_get_layer(edge.src_node, topo_layer)
            topo_layer[node] = max(topo_layer[node], topo_layer[edge.src_node] + 1)
    topo_layer = {}
    for node in output_node_list:
        topo_sort_get_layer(node, topo_layer)

    def topo_sort_dfs(node, visited, topo_order):
        if node in visited:
            return
        visited.add(node)
        ordered_input_nodes = sorted([edge.src_node for edge in node.inputs], key=lambda n:topo_layer[n], reverse=True)
        for n in ordered_input_nodes:
            topo_sort_dfs(n, visited, topo_order)
        topo_order.append(node)
    visited = set()
    topo_order = []
    for node in output_node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def find_topo_sort(output_node_list):
    def topo_sort_dfs(node, visited, topo_order):
        if node in visited:
            return
        visited.add(node)
        for edge in node.inputs:
            topo_sort_dfs(edge.src_node, visited, topo_order)
        topo_order.append(node)
    visited = set()
    topo_order = []
    for node in output_node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order
