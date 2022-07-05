from memopt.graph import Edge, OutputNode, find_topo_sort
from memopt.fusion import DefaultPolicy
from memopt.utils import CompileResult, compile_and_load_parallel, compose_global_kernel
from memopt.reference import get_subgraph_reference_outputs
from memopt import get_log_level
import numpy as np
import hashlib
import traceback
import sys

_cache_store = {}

def get_cache(sig):
    return _cache_store[sig]

def count_cache(sig):
    return sig in _cache_store

def set_cache(sig, value):
    global _cache_store
    _cache_store[sig] = value

def get_max_diff(tensor_list_a, tensor_list_b):
    assert len(tensor_list_a) > 0
    total_diff = [0]
    for a, b in zip(tensor_list_a, tensor_list_b):
        assert a.shape == b.shape
        diff = np.abs(a-b)
        diff = diff / np.abs(b).clip(1) # handle large floating numbers
        diff = np.max(diff)
        total_diff.append(diff)
    total_diff = max(total_diff)
    return total_diff

def subgraph_hash(output_nodes):
    nodes = find_topo_sort(output_nodes)
    node_hash, edge_hash = [], []
    node_idx = {node : i for i, node in enumerate(nodes)}
    for node in nodes:
        if node.is_placeholder():
            value = "placeholder"
        elif node.is_output():
            value = "output"
        else:
            value = node.ir
        hex = hashlib.sha1(bytes(value, encoding="utf-8")).hexdigest()
        node_hash.append(int(hex, 16))
        for edge in node.outputs:
            sig = (node_idx[edge.src_node], node_idx[edge.dst_node], edge.src_id, edge.dst_id)
            edge_hash.append(hash(sig))
    graph_sig = hash(tuple(node_hash + edge_hash))
    return graph_sig

def _extract_subgraph(nodes):
    node_map = {}
    output_nodes = []
    for node in nodes:
        input_list = []
        for edge in node.inputs:
            if edge.src_node in nodes:
                input_list.append((node_map[edge.src_node], edge.src_id))
            else:
                input_list.append(None)
        node_map[node] = node.clone(input_list)
        global_output = set()
        for edge in node.outputs:
            if not edge.dst_node in nodes:
                global_output.add(edge.src_id)
        for idx in global_output:
            output_nodes.append(OutputNode(node_map[node] ,idx))

    # get subgraph inputs and outputs mapping to the original nodes
    # this should get the same results as final code generation
    ordered_nodes = find_topo_sort(output_nodes)
    new2old = {v: k for k, v in node_map.items()}
    input_desc, output_desc = [], []
    for node in ordered_nodes:
        if node.is_placeholder():
            origin_node = new2old[node.outputs[0].dst_node]
            x = origin_node.args[node.outputs[0].dst_id].name
            assert(x.startswith("input"))
            dst_id = int(x[5:]) # node.outputs[0].dst_id is not the same with dst_id, since some inputs might be unused and removed
            input_desc.append([origin_node.name, dst_id])
        elif node.is_output():
            output_desc.append([new2old[node.inputs[0].src_node].name, node.inputs[0].src_id])

    return output_nodes, input_desc, output_desc

def eliminate_memcpy(output_nodes):
    nodes = find_topo_sort(output_nodes)
    eliminated_node_cnt = 0
    for node in nodes:
        if node.get_tag("memcpy") and len(node.inputs) == 1 and len(node.outputs) == 1:
            inode = node.inputs[0].src_node
            onode = node.outputs[0].dst_node
            inode_id = node.inputs[0].src_id
            onode_id = node.outputs[0].dst_id
            if inode.is_placeholder() and not onode.is_output():
                inode.set_shape(node.get_shape(), overwrite=True)
                edge = Edge(inode, onode, inode_id, onode_id)
                inode.set_outputs(inode_id, edge)
                onode.set_inputs(onode_id, edge)
                eliminated_node_cnt += 1
            elif not inode.is_placeholder() and onode.is_output():
                onode.set_shape(inode.get_shape(inode_id), overwrite=True)
                edge = Edge(inode, onode, inode_id, onode_id)
                inode.set_outputs(inode_id, edge)
                onode.set_inputs(onode_id, edge)
                eliminated_node_cnt += 1
    if eliminated_node_cnt > 0:
        eliminate_memcpy(output_nodes)

def tune(nodes, arch, device="cuda:0", kernel_name="Fused", topk=10, check=True):
    if get_log_level() >= 1:
        print("Tuning", [node.name for node in nodes])
    output_nodes, input_desc, output_desc = _extract_subgraph(nodes)
    eliminate_memcpy(output_nodes)

    signature = subgraph_hash(output_nodes)
    if count_cache(signature):
        print("Found in cache")
        cached = get_cache(signature)
        if cached is None:
            return None
        result = CompileResult(cached.config, cached.code.replace(cached.name, kernel_name),
            cached.block_size, cached.grid_size, kernel_name, cached.args)
        result.latency = cached.latency
        result.set_io_desc(input_desc, output_desc)
        result.origin = cached
        return result

    policy = DefaultPolicy(output_nodes, arch)
    if any([node.get_tag("skip") for node in policy.ordered_nodes]):
        return None
    configs = policy.emit_config(topk)
    if len(configs) == 0:
        return None
    compile_results = []
    for config in configs:
        try:
            cpresult = compose_global_kernel(output_nodes, config, "cuda", name=kernel_name)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            continue
        cpresult.append_host_call()
        cpresult.set_io_desc(input_desc, output_desc)
        compile_results.append(cpresult)
    compile_and_load_parallel(compile_results)
    values = []
    for cpresult in compile_results:
        if get_log_level() >= 2: print(cpresult.config)
        if cpresult.lib is None:
            latency = 10000
        else:
            latency = cpresult.profile(device)
        values.append(latency)
        if get_log_level() >= 2: print(latency)
    compile_results = list(filter(lambda x:x.latency<10000, compile_results))
    compile_results = sorted(compile_results, key=lambda x:x.latency)
    if len(compile_results) == 0:
        return None

    if get_log_level() >= 2:
        print("Best Config:", compile_results[0].config)
    if get_log_level() >= 1:
        print("top1: {} \ttopk: {}".format(values[0], min(values)), flush=True)
    if not check:
        set_cache(signature, compile_results[0])
        return compile_results[0]

    ref_out = get_subgraph_reference_outputs(output_nodes, device=device)
    for best in compile_results:
        out = best.get_example_outputs(device)
        total_diff = get_max_diff(out, ref_out)
        if get_log_level() >= 1: print("Diff:", total_diff)
        if total_diff > 1e-3:
            continue
        set_cache(signature, best)
        return best
    set_cache(signature, None)
    return None
