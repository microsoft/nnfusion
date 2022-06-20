from .graph import find_topo_sort, topo_order
from tvm import te
import tvm
import numpy as np
import torch

def get_ref_tensor(shape:list, device:str, dtype:str):
    dtype = torch.__getattribute__(str(dtype))
    if dtype.is_floating_point:
        return torch.empty(*shape, device=device, dtype=dtype).uniform_(0.1, 1.0)
    else:
        return torch.ones(*shape, device=device, dtype=dtype)

def get_subgraph_reference_outputs(output_nodes, device="cuda:0", seed=0):
    topo_order = find_topo_sort(output_nodes)
    torch.cuda.set_device(device)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    values = {}
    for node in topo_order:
        if node.is_placeholder():
            shape = list(map(int, node.get_shape()))
            arr = get_ref_tensor(shape, device, node.get_dtype())
            arr = tvm.nd.array(arr.cpu().numpy())
            values[(node, 0)] = arr
    ordered_nodes = list(filter(
        lambda n: not n.is_placeholder() and not n.is_output(),
        find_topo_sort(output_nodes)
    ))
    for node in ordered_nodes:
        schedule = node.create_schedule()
        mod = tvm.build(schedule, node.args, target="llvm")
        args = []
        for edge in node.inputs:
            args.append(values[(edge.src_node, edge.src_id)])
        for idx in range(len(node.inputs), len(node.args)):
            arr = tvm.nd.empty(node.args[idx].shape, node.args[idx].dtype)
            values[(node, idx - len(node.inputs))] = arr
            args.append(arr)
        mod(*args)
    results = []
    for node in output_nodes:
        edge = node.inputs[0]
        tensor = values[(edge.src_node, edge.src_id)]
        results.append(tensor.numpy())
    return results
