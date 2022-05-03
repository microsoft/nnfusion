from policy import ConstructionPolicyV2

from arch import V100

arch = V100()

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
    node_id = 0
    def __init__(self, inputs, name):
        self.node_id = Node.node_id
        Node.node_id += 1
        self.name = name
        self._out_edges = []
        self._in_edges = []
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

    def __repr__(self) -> str:
        return "<Node, " + self.name + ">"

class PlaceHolderNode(Node):
    def __init__(self, name):
        super().__init__([], "PlaceHolder " + name)

class OutputNode(Node):
    def __init__(self, node, id=0):
        super().__init__([(node, id)], "Output ")

class MatMulNode(Node):
    def __init__(self, inputs, n, m ,k):
        super().__init__(inputs, "MatMul")
        from op import MatmulOp
        from .tvm_ops import tvm_matmul
        self.op = MatmulOp(m, k, n)
        self.args = tvm_matmul(n, m, k)

    def emit_config(self):
        stage = self.sch[self.args[2]]
        saxis_names = [axis.var.name for axis in stage.op.axis]
        raxis_names = [axis.var.name for axis in stage.op.reduce_axis]
        policy = ConstructionPolicyV2(self.op, arch, saxis_names, raxis_names)
        return policy.emit_config_without_trails(10)[:10]

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

class ComputeNode(Node):
    def __init__(self, inputs, args):
        super().__init__(inputs, "Compute")
        self.args = args

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
