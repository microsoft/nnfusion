import io
from typing import Dict, List

import numpy as np

from .bestfit import BestFit
from .config import Config
from .graph import Node, find_topo_sort
from .schedule import schedule
from .tvm_build import _type_map, tvm_build
from .utils import CompileResult


class CodeGenerator():
    def __init__(self) -> None:
        pass

    def _get_tensor_name_map(self):
        tensor_name_map = {}
        num_inputs, num_outputs = 0, 0
        for op in self.topo_order:
            if op.is_placeholder():
                tensor_name_map[op] = "input" + str(num_inputs)
                num_inputs += 1
            elif op.is_output():
                tensor_name_map[op] = "output" + str(num_outputs)
                num_outputs += 1
        global_args_name_map = {}
        for op in self.topo_order:
            if op.is_placeholder() or op.is_output():
                continue
            else:
                for edge in op.inputs:
                    if edge.src_node.is_placeholder():
                        global_args_name_map[op.args[edge.dst_id]] = tensor_name_map[edge.src_node]
                for edge in op.outputs:
                    if edge.dst_node.is_output():
                        global_args_name_map[op.args[edge.src_id+len(op.inputs)]] = tensor_name_map[edge.dst_node]
        return global_args_name_map

    def _write_code(self, kernel_codes, statements, global_args_name_map):
        if self.allocator.limit > 0:
            statements.insert(0, "__shared__ char shared[{}];".format(self.allocator.limit))
        else:
            statements.insert(0, "char* shared = NULL;")
        kernel_args_dtype_map = {v : _type_map[k.dtype] for k, v in global_args_name_map.items()}
        kernel_args_name = sorted(set(global_args_name_map.values()), key=self._arg_name_cmp)
        kernel_args = ["{}* {}".format(kernel_args_dtype_map[arg], arg) for arg in kernel_args_name]
        prefix = "__global__ void __launch_bounds__({}) {}({})".format(
            np.prod(self.block_size), self.kernel_name, ", ".join(kernel_args))
        code = io.StringIO()
        [code.write(kernel_code) for kernel_code in kernel_codes]
        code.write(prefix)
        code.write(" {\n")
        [code.write("  "+stmt+"\n") for stmt in statements]
        code.write("}\n")
        return code.getvalue()

    def _arg_name_cmp(self, arg_name):
        if arg_name.startswith("input"):
            return (0, int(arg_name[5:]))
        elif arg_name.startswith("output"):
            return (1, int(arg_name[6:]))
        else:
            assert(False)

    def _can_free(self, node, out_id):
        for edge in node.outputs:
            if edge.src_id == out_id and edge.dst_node not in self.done_ops:
                return False
        return True

    def compile(self, output_nodes: List[Node], configs: Dict[Node, Config], target: str, kernel_name: str) -> CompileResult:
        self.kernel_name = kernel_name
        self.topo_order = find_topo_sort(output_nodes) # List[Node]
        self.target = target
        compute_nodes = list(filter(lambda node: not(node.is_output() or node.is_placeholder()), self.topo_order))
        if len(compute_nodes) == 1:
            return self._compile_single_node(compute_nodes[0], configs[compute_nodes[0]])
        global_args_name_map = self._get_tensor_name_map() # {Tensor : "input0"}

        self.allocator = BestFit()
        self.block_size, self.grid_size = None, None
        self.done_ops = set()
        block_map = {} # {(Node, output_id) : Block}
        strides_map = {} # {(Node, output_id) : Block}
        statements, kernel_codes = [], []

        for op in self.topo_order:
            self.done_ops.add(op)
            if op.is_placeholder() or op.is_output():
                continue
            config = configs[op]
            shared_inputs_idx = [edge.dst_id for edge in filter(
                lambda edge : not edge.src_node.is_placeholder(), op.inputs)]
            shared_outputs_idx = list({edge.src_id + len(op.inputs) for edge in filter(
                lambda edge : not edge.dst_node.is_output(), op.outputs)}) # use set, may have multiple consumers
            shared_inputs = [op.args[idx] for idx in shared_inputs_idx]
            shared_inputs_strides = {}
            for idx in shared_inputs_idx:
                if (op.inputs[idx].src_node, op.inputs[idx].src_id) in strides_map:
                    shared_inputs_strides[op.args[idx]] = strides_map[(op.inputs[idx].src_node, op.inputs[idx].src_id)]

            sch = schedule(op.args, config, shared_inputs=shared_inputs,
                shared_inputs_strides=shared_inputs_strides, shared_outputs=shared_outputs_idx)

            # Some inputs which will be used later cannot be overwritten by other internal shared memory,
            # so we must put these tensor in reuse_disabled_inputs.
            # Inputs that will be freed after this kernel can be overwritten and reused in this kernel.
            reuse_disabled_inputs = [op.args[idx] for idx in filter(
                lambda idx: not self._can_free(op.inputs[idx].src_node, op.inputs[idx].src_id), shared_inputs_idx)]
            # generate the kernel code for this node
            func_name = "_".join([self.kernel_name, str(len(statements)), op.name]) # unique globally
            kernel_code, exteral_shared_memroy_size, total_internal_shared_memory = tvm_build(
                sch, self.target,
                name=func_name,
                global_kernel=False,
                reuse_disabled_inputs=reuse_disabled_inputs,
            )
            for idx in config.output_strides:
                strides_map[(op, idx-len(op.inputs))] = config.output_strides[idx]
            kernel_codes.append(kernel_code)
            if self.block_size is None:
                self.block_size, self.grid_size = sch.block_size, sch.grid_size
            else:
                assert(self.block_size == sch.block_size and self.grid_size == sch.grid_size)
            for idx, tensor in zip(shared_inputs_idx, shared_inputs):
                num_bytes = exteral_shared_memroy_size[tensor]
                src_node, src_id = op.inputs[idx].src_node, op.inputs[idx].src_id
                block = block_map[(src_node, src_id)]
                if block.size() < num_bytes:
                    raise Exception("Shared memory mismatched", op, tensor, block.size(), num_bytes)

            # make memory plan
            internal_shared_mem = self.allocator.malloc(total_internal_shared_memory)
            for idx in shared_inputs_idx:
                if op.args[idx] not in reuse_disabled_inputs:
                    src_node, src_id = op.inputs[idx].src_node, op.inputs[idx].src_id
                    self.allocator.free(block_map[(src_node, src_id)])
            self.allocator.free(internal_shared_mem)
            for idx in shared_outputs_idx:
                num_bytes = exteral_shared_memroy_size[idx]
                block_map[(op, idx-len(op.inputs))] = self.allocator.malloc(num_bytes)

            # generate kernel call statement
            arg_list = []
            for idx in range(len(op.args)):
                dtype = _type_map[op.args[idx].dtype]
                if idx in shared_inputs_idx:
                    src_node, src_id = op.inputs[idx].src_node, op.inputs[idx].src_id
                    arg_list.append(f"({dtype}*)(shared+{block_map[(src_node, src_id)].start})")
                elif idx in shared_outputs_idx:
                    arg_list.append(f"({dtype}*)(shared+{block_map[(op, idx-len(op.inputs))].start})")
                else:
                    arg_list.append(global_args_name_map[op.args[idx]])
            arg_list.append(f"shared+{internal_shared_mem.start}")
            statements.append(func_name + "(" + ", ".join(arg_list) + ");")

        code = self._write_code(kernel_codes, statements, global_args_name_map)

        # fused kernel args
        global_args_names = sorted(set(global_args_name_map.values()), key=self._arg_name_cmp)
        global_args_name_rmap = {v : k for k, v in global_args_name_map.items()}
        global_args = [global_args_name_rmap[x] for x in global_args_names]

        return CompileResult(configs, code, self.block_size, self.grid_size, self.kernel_name, global_args)

    def _compile_single_node(self, node: Node, config: Config):
        sch = schedule(node.args, config)
        self.block_size, self.grid_size = sch.block_size, sch.grid_size
        code, _, _ = tvm_build(sch, self.target, name=self.kernel_name, global_kernel=True, flatten_block=False)
        return CompileResult({node: config}, code, self.block_size, self.grid_size, self.kernel_name, node.args)
