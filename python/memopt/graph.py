import tvm
import lang
import numpy as np

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

    def get_shape(self, id=0):
        return self._shapes[id]

    def set_shape(self, shape, id=0):
        if len(self._shapes) <= id:
            self._shapes.extend([None for _ in range(id - len(self._shapes) + 1)])
        elif self._shapes[id] is not None:
            assert self._shapes[id] == list(map(int, shape)), (self._shapes, list(map(int, shape)))
        self._shapes[id] = list(map(int, shape))

    def get_dtype(self, id=0):
        return self._dtypes[id]

    def set_dtype(self, dtype, id=0):
        if len(self._dtypes) <= id:
            self._dtypes.extend([None for _ in range(id - len(self._dtypes) + 1)])
        elif self._dtypes[id] is not None:
            assert self._dtypes[id] == str(dtype), (self._dtypes, str(dtype))
        self._dtypes[id] = str(dtype)

    def is_placeholder(self):
        return False

    def is_output(self):
        return False

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
        self.args = self._input_args + self._output_args
        self.ana = lang.get_analyzer_by_ir(self.ir)
        assert len(self.inputs) + 1 == len(self.args)
        for edge, arg in zip(self.inputs, self.args):
            edge.src_node.set_shape(arg.shape, edge.src_id)
            edge.src_node.set_dtype(arg.dtype, edge.src_id)
        for output_id, arg in enumerate(self._output_args):
            self.set_shape(arg.shape, output_id)
            self.set_dtype(arg.dtype, output_id)
        self._extract_axis()
        self.reduction_inputs = self.ana.get_reduction_inputs()

    def infer_dependency(self, shape, rstep={}):
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
        shapes = self.ana.infer(shape, rstep)
        for tensor in self.reduction_inputs:
            if tensor.startswith("input"):
                src_node = self.inputs[int(tensor[5:])].src_node
                if not src_node.is_placeholder():
                    continue
            result += np.prod(shapes[tensor]) * 4 # TODO : Add data type
        return result

    def infer_reduction_inputs(self, shape, rstep):
        result = 0
        shapes = self.ana.infer(shape, rstep)
        for tensor in self.reduction_inputs:
            result += np.prod(shapes[tensor])
        return result

    # axis name -> axis length
    def _extract_axis(self):
        queue = self._output_args.copy()
        self.raxis = {}
        visited = {}
        while len(queue) > 0:
            t = queue.pop(0)
            visited[t] = True
            if isinstance(t.op, tvm.te.PlaceholderOp):
                continue
            for axis in t.op.reduce_axis:
                assert(str(axis.var.name) not in self.raxis), axis.var.name
                self.raxis[str(axis.var.name)] = int(axis.dom.extent)
            for it in t.op.input_tensors:
                if not it in visited:
                    queue.append(it)

        self.saxis = {}
        for axis in self._output_args[0].op.axis:
            assert(str(axis.var.name) not in self.saxis), axis.var.name
            self.saxis[str(axis.var.name)] = int(axis.dom.extent)

    def create_schedule(self):
        return tvm.te.create_schedule([x.op for x in self._output_args])

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
