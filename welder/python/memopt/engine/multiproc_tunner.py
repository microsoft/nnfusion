import os
import tvm

from tvm.contrib.popen_pool import PopenPoolExecutor

from ..code_generator import CodeGenerator
from ..graph import find_topo_sort
from ..utils import CompileResult
from .base_tunner import Tunner, _extract_subgraph, eliminate_memcpy
from .load_model import load_model


class _save:
    pass

def init_server(path):
    ordered_nodes = load_model(path)
    _save.node_map = {node.name: node for node in ordered_nodes}

def call_build(node_names, send_config, kernel_name, target_str):
    cgen = CodeGenerator()
    nodes = [_save.node_map[name] for name in node_names]
    output_nodes, _, _ = _extract_subgraph(nodes)
    eliminate_memcpy(output_nodes)
    config = {}
    for node in find_topo_sort(output_nodes):
        if node.name in send_config:
            config[node] = send_config[node.name]
    try:
        cpresult = cgen.compile(output_nodes, config, tvm.target.Target(target_str), kernel_name=kernel_name)
    except Exception as e:
        # traceback.print_exc(file=sys.stdout)
        return e
    return [cpresult.code, cpresult.block_size, cpresult.grid_size, cpresult.args]

class MultiProcTunner(Tunner):
    def __init__(self, input_file_path, arch, device="cuda:0", check=False, topk=10) -> None:
        super().__init__(arch, device, check, topk)
        num_procs = min(topk, os.cpu_count(), 10)
        self.pool = PopenPoolExecutor(max_workers=num_procs, timeout=None, initializer=init_server, initargs=[input_file_path])

    def generate_code(self, output_nodes, configs, kernel_name):
        compile_results = []
        node_names = [node.name for node in self.current_nodes]
        futures = []
        for config in configs:
            send_config = {node.name : config[node] for node in config}
            futures.append(self.pool.submit(call_build, node_names, send_config, kernel_name, str(self.arch.target)))
        for future, config in zip(futures, configs):
            result = future.result()
            if isinstance(result, Exception):
                print(result)
                continue
            code, block_size, grid_size, args = result
            compile_results.append(CompileResult(config, code, block_size, grid_size, kernel_name, args))
        return compile_results
