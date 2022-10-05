from typing import List

import numpy as np

from ..arch import Arch
from ..bestfit import BestFit
from ..config import Config, TileDict
from ..graph import Node
from .common import factorize, get_all_factors
from .default import DefaultPolicy


class TCPolicy(DefaultPolicy):
    def __init__(self, output_nodes: List[Node], arch: Arch) -> None:
        super().__init__(output_nodes, arch)
        self.wmma_k = 16
        tc_nodes = list(filter(lambda node: node.get_tag("tensorCoreConfig"), self.ordered_nodes))
        for node in tc_nodes:
            for k in node.raxis:
                assert node.raxis[k] % self.wmma_k == 0

    def _compute_shared_memory_usage(self, td):
        allocator = BestFit()
        block_map = {}
        processed = set()
        def can_free(node, out_id):
            for edge in node.outputs:
                if edge.src_id == out_id and edge.dst_node not in processed:
                    return False
            return True
        for node in self.ordered_nodes:
            if node.get_tag("tensorCoreConfig"):
                node_internal_bytes = node.infer_smem_usage_TensorCore(td.get_tile(node), td.get_rstep(node))
            else:
                node_internal_bytes = node.infer_smem_usage(td.get_tile(node), td.get_rstep(node))
            block = allocator.malloc(node_internal_bytes)
            allocator.free(block)
            # free inputs
            processed.add(node)
            for edge in node.inputs:
                if not edge.src_node.is_placeholder() and can_free(edge.src_node, edge.src_id):
                    allocator.free(block_map.pop((edge.src_node, edge.src_id)))
            # alloc outputs
            for edge in node.outputs:
                if not edge.dst_node.is_output() and (node, edge.src_id) not in block_map:
                    dtype_bytes = node.get_dtype(edge.src_id).bits // 8
                    stride = td.stride_map[node][len(node.inputs) + edge.src_id]
                    output_elem = stride.compute_elements_from_shape(td.get_tile(node))
                    block_map[(node, edge.src_id)] = allocator.malloc(output_elem * dtype_bytes)

        assert len(block_map) == 0
        return allocator.limit

    def get_node_reduce_step_candidates(self, node):
        if not node.get_tag("tensorCoreConfig"):
            return super().get_node_reduce_step_candidates(node)
        else:
            # must be a a multiple of wmma_k
            return {k : [x * self.wmma_k for x in get_all_factors(node.raxis[k] // self.wmma_k)] for k in node.raxis}

    def check_tile_shape_isvalid(self, td: TileDict):
        out_node = self.output_nodes[0]
        grid_size = np.prod([np.ceil(y / x) for x, y in zip(td.output_tile, out_node.get_shape())])
        for node in self.ordered_nodes:
            node_grid_size = np.prod([np.ceil(y / x) for x, y in zip(td.tile_map[node], node.get_shape())])
            if node_grid_size != grid_size:
                return False
            if node.get_tag("tensorCoreConfig"):
                ax_m, ax_n = node.get_tag("tensorCoreConfig")
                CS_m = td.tile_map[node][ax_m]
                CS_n = td.tile_map[node][ax_n]
                wmma_invalid = [CS_m % wmma_m or CS_n % wmma_n for wmma_m, wmma_n in [(16, 16), (8, 32), (32, 8)]]
                if all(wmma_invalid):
                    return False
                if any([y % x for x, y in zip(td.tile_map[node], node.get_shape())]):
                    return False
        return True

    def compute_stride_map(self, td: TileDict):
        super().compute_stride_map(td)
        for node in self.ordered_nodes:
            if not node.get_tag("tensorCoreConfig"): continue
            A_stride, B_stride, C_stride = node.infer_strides_TensorCore(td.get_tile(node))
            for edge in node.outputs:
                td.stride_map[node][edge.src_id+len(node.inputs)] = C_stride
            deps = node.infer_dependency_reduce_inputs(td.get_tile(node))
            for name, stride in zip(deps.keys(), [A_stride, B_stride]):
                if name.startswith("input"):
                    input_id = [arg.name for arg in node._input_args].index(name)
                    assert(input_id >= 0)
                    src_id, src_node = node.inputs[input_id].src_id, node.inputs[input_id].src_node
                    if not src_node.is_placeholder():
                        td.stride_map[src_node][int(src_id + len(src_node.inputs))] = stride

    def _assign_block_size(self, node: Node, tile, rsteps, block_size):
        if not node.get_tag("tensorCoreConfig"):
            return super()._assign_block_size(node, tile, rsteps, block_size)
        ax_m, ax_n = node.get_tag("tensorCoreConfig")
        assert ax_m is not None and ax_n is not None
        if block_size % 32 != 0:
            return None
        warps = block_size // 32
        ndim = len(tile)
        if tile[ax_m] > tile[ax_n]:
            wmma = [32, 8, 16]
        elif tile[ax_m] < tile[ax_n]:
            wmma = [8, 32, 16]
        else:
            wmma = [16, 16, 16]
        wmma_tile = [1 for i in range(ndim)]
        wmma_tile[ax_m] = wmma[0]
        wmma_tile[ax_n] = wmma[1]
        space = [tile[i] // wmma_tile[i] for i in range(ndim)]
        if tile[ax_m] % wmma_tile[ax_m] != 0 or tile[ax_n] % wmma_tile[ax_n]:
            return None
        if np.prod(space) % warps != 0:
            return None
        factors = factorize(np.prod(space) // warps)

        def _score(node, thread): # small is better
            score = 0
            block_tile = [int(np.ceil(tile[i] / thread[i])) for i in range(ndim)]
            shape = node.infer_dependency(block_tile)
            for edge in node.inputs:
                score += np.prod(shape[edge.dst_id]) / self.arch.memory_bw(1)
            return score

        warp_tile = wmma_tile.copy()
        for factor in reversed(factors):
            score_map = {}
            for i in range(ndim):
                if tile[i] % (warp_tile[i] * factor) != 0:
                    continue
                warp_tile[i] *= factor
                score_map[i] = (_score(node, warp_tile), i)
                warp_tile[i] //= factor
            if len(score_map) == 0:
                return None
            dim_order = sorted(score_map.keys(), key=lambda x:score_map[x])
            warp_tile[dim_order[0]] *= factor

        codegen_dict = Config()
        codegen_dict.use_tc = True
        codegen_dict.block = tile
        codegen_dict.warp = warp_tile
        codegen_dict.rstep = [int(rsteps[ax]) for ax in node.raxis]
        codegen_dict.wmma = wmma
        codegen_dict.complete_config(node)
        return codegen_dict
