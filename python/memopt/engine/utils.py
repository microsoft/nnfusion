from tvm import te

from ..graph import IRNode, find_topo_sort
from ..te_utils import connect_tensor_graph


def _inline_node(A: IRNode, B: IRNode):
    connection = {}
    A_inputs = {(edge.src_node, edge.src_id): arg for edge, arg in zip(A.inputs, A.args)}
    # get the connection for A and B
    for edge in B.inputs:
        if edge.src_node is A:
            connection[B.args[edge.dst_id]] = A.args[edge.src_id + len(A.inputs)]
        elif (edge.src_node, edge.src_id) in A_inputs:
            connection[B.args[edge.dst_id]] = A_inputs[(edge.src_node, edge.src_id)]

    args = connect_tensor_graph(A.args, B.args, connection)
    num_inputs = sum([isinstance(arg.op, te.PlaceholderOp) for arg in args])
    inputs = [None for _ in range(num_inputs)]
    name = "__".join([A.name, B.name])
    C = IRNode(inputs, args, name)
    for edge in A.outputs:
        # supports all outputs of A connect to B for now
        assert edge.dst_node is B
    for edge in B.outputs:
        edge.src_node = C
        C.outputs.append(edge)
    input_id = 0
    for edge in A.inputs:
        edge.dst_node = C
        edge.dst_id = input_id
        C.set_inputs(input_id, edge)
        input_id += 1
    for i, edge in enumerate(B.inputs):
        if B.args[i] in connection:
            # B might share inputs with A, we need to remove redundant edges
            edge.src_node.outputs.remove(edge)
            continue
        edge.dst_node = C
        edge.dst_id = input_id
        C.set_inputs(input_id, edge)
        input_id += 1
    for k, v in A._tag.items():
        C.add_tag(k, v)
    for k, v in B._tag.items():
        C.add_tag(k, v)
    if C.reduce_op:
        C.schedule_stage = C.reduce_op
    return C

def insert_local_connections(output_nodes, connections):
    mapping = {node.name: node for node in find_topo_sort(output_nodes)}
    replacement_map = {}
    for A, B in connections:
        if A in replacement_map:
            A = replacement_map[A]
        if B in replacement_map:
            B = replacement_map[B]
        new_node = _inline_node(mapping[A], mapping[B])
        replacement_map[A] = new_node.name
        replacement_map[B] = new_node.name