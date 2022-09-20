from .default import DefaultPolicy
from .common import factorize, get_all_factors
from .config import Config
from arch.Arch import Arch
from memopt.graph import Node
from memopt.bestfit import BestFit

from typing import List, Dict
import numpy as np

class TCPolicy(DefaultPolicy):
    def __init__(self, output_nodes: List[Node], arch: Arch) -> None:
        super().__init__(output_nodes, arch)
        self.wmma_k = 16
        tc_nodes = list(filter(lambda node: node.get_tag("tensorCoreConfig"), self.ordered_nodes))
        for node in tc_nodes:
            for k in node.raxis:
                assert node.raxis[k] % self.wmma_k == 0

    def _compute_shared_memory_usage(self, tile_map, rstep_map):
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
                node_internal_bytes, output_elem = node.infer_smem_usage_TensorCore(tile_map[node], rstep_map[node])
            else:
                node_internal_bytes = node.infer_smem_usage(tile_map[node], rstep_map[node])
                output_elem = np.prod(tile_map[node])
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
                    block_map[(node, edge.src_id)] = allocator.malloc(output_elem * dtype_bytes)

        assert len(block_map) == 0
        return allocator.limit

    def get_node_reduce_step_candidates(self, node):
        if not node.get_tag("tensorCoreConfig"):
            return super().get_node_reduce_step_candidates(node)
        else:
            # must be a a multiple of wmma_k
            return {k : [x * self.wmma_k for x in get_all_factors(node.raxis[k] // self.wmma_k)] for k in node.raxis}

    def check_tile_shape_isvalid(self, out_tile):
        output_tile_map = self.get_tile_map(out_tile)
        _, tile_map = self._compute_memory_footprint(output_tile_map)
        out_node = self.output_nodes[0]
        grid_size = np.prod([np.ceil(y / x) for x, y in zip(out_tile, out_node.get_shape())])
        for node in self.ordered_nodes:
            node_grid_size = np.prod([np.ceil(y / x) for x, y in zip(tile_map[node], node.get_shape())])
            if node_grid_size != grid_size:
                return False
            if node.get_tag("tensorCoreConfig"):
                ax_m, ax_n = node.get_tag("tensorCoreConfig")
                CS_m = tile_map[node][ax_m]
                CS_n = tile_map[node][ax_n]
                wmma_invalid = [CS_m % wmma_m or CS_n % wmma_n for wmma_m, wmma_n in [(16, 16), (8, 32), (32, 8)]]
                if all(wmma_invalid):
                    return False
                if any([y % x for x, y in zip(tile_map[node], node.get_shape())]):
                    return False
        return True

    def _assign_block_size(self, node: Node, tile, rstep_map, block_size):
        if not node.get_tag("tensorCoreConfig"):
            return super()._assign_block_size(node, tile, rstep_map, block_size)
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
        codegen_dict.rstep = [int(rstep_map[ax]) for ax in node.raxis]
        codegen_dict.wmma = wmma
        codegen_dict.complete_config(node)
        return codegen_dict
