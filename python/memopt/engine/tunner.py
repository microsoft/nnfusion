from memopt.graph import OutputNode, find_topo_sort
from memopt.fusion import DefaultPolicy
from memopt.utils import compile_and_load_parallel, compose_global_kernel
from memopt.reference import get_subgraph_reference_outputs
from memopt import get_log_level
import numpy as np

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
            input_desc.append([new2old[node.outputs[0].dst_node].name, node.outputs[0].dst_id])
        elif node.is_output():
            output_desc.append([new2old[node.inputs[0].src_node].name, node.inputs[0].src_id])

    return output_nodes, input_desc, output_desc

def tune(nodes, arch, kernel_name="Fused", topk=10, check=True):
    if get_log_level() >= 1:
        print("Start fusing ", [node.name for node in nodes])
    output_nodes, input_desc, output_desc = _extract_subgraph(nodes)
    policy = DefaultPolicy(output_nodes, arch)
    if any([node.get_tag("skip") for node in policy.ordered_nodes]):
        return None
    configs = policy.emit_config(topk)
    if len(configs) == 0:
        return None
    compile_results = []
    for config in configs:
        cpresult = compose_global_kernel(output_nodes, config, "cuda", name=kernel_name)
        cpresult.append_host_call()
        compile_results.append(cpresult)
    compile_and_load_parallel(compile_results)
    best_latency = 10000
    best = None
    values = []
    for cpresult in compile_results:
        if get_log_level() >= 2: print(cpresult.config)
        if cpresult.lib is None:
            latency = 10000
        else:
            latency = cpresult.profile()
        values.append(latency)
        if latency < best_latency:
            best_latency = latency
            best = cpresult
        if get_log_level() >= 2: print(latency)
    if best is not None:
        best.set_io_desc(input_desc, output_desc)
        if get_log_level() >= 1:
            print("Best Config:", best.config)
            print("top1: {} \ttop10: {}".format(values[0], min(values)))
        if check:
            total_diff = 0
            out = best.get_example_outputs()
            ref_out = get_subgraph_reference_outputs(output_nodes)
            for a, b in zip(out, ref_out):
                diff = np.max(np.abs(a-b))
                total_diff += diff
            if get_log_level() >= 1: print("Diff:", total_diff)
            if total_diff > 1e-3:
                best = None
        if get_log_level() >= 1:
            print("-" * 80, flush=True)
    return best
