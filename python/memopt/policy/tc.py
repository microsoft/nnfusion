from typing import Dict, List, Tuple

import numpy as np

from ..arch import Arch
from ..config import Config, Stride, TileDict
from ..graph import IRNode, Node
from .common import factorize, get_all_factors
from .default import DefaultPolicy


class TCPolicy(DefaultPolicy):
    def __init__(self, output_nodes: List[Node], arch: Arch) -> None:
        super().__init__(output_nodes, arch)
        self.wmma_k = 16

    def _compute_tc_strides(self, node: IRNode, tile: List[int], rstep: Dict[str, int]={}) -> Tuple[Stride, Stride, Stride]:
        shapes = node.infer_dependency_reduce_inputs(tile, rstep)
        AS_shape, BS_shape = shapes.values()
        CS_shape = tile
        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = node.infer_tensorcore_axis()
        # applying strides
        offset = 8
        A_high_ax = min(A_ax_m, A_ax_k)
        B_high_ax = min(B_ax_n, B_ax_k)
        C_high_ax = min(C_ax_m, C_ax_n)
        A_stride = Stride(stride=np.prod(AS_shape[A_high_ax+1:]) + offset, ax=A_high_ax)
        B_stride = Stride(stride=np.prod(BS_shape[B_high_ax+1:]) + offset, ax=B_high_ax)
        C_stride = Stride(stride=np.prod(CS_shape[C_high_ax+1:]) + offset, ax=C_high_ax)
        return A_stride, B_stride, C_stride

    def _use_cutlass_mma(self, node: IRNode, td: TileDict):
        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = node.infer_tensorcore_axis()
        tile = td.get_tile(node)
        use_cutlass_warp_mma = True
        use_cutlass_warp_mma &= tile[C_ax_m] % self.arch.cutlass_mma[0] == 0
        use_cutlass_warp_mma &= tile[C_ax_n] % self.arch.cutlass_mma[1] == 0
        # cutlass_warp_mma currently don't support shared inputs as it uses pipeline approaches
        use_cutlass_warp_mma &= all([edge.src_node.is_placeholder() for edge in node.inputs])
        # use pipeline for large reduce ops
        use_cutlass_warp_mma &= all([x > 64 for x in node.raxis.values()])
        # cutlass_warp_mma don't support batched mm inside a block
        for idx, value in enumerate(tile):
            if idx not in [C_ax_m, C_ax_n]: use_cutlass_warp_mma &= value==1
        return use_cutlass_warp_mma

    def _can_implement_layout(self, node: IRNode, td: TileDict):
        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = node.infer_tensorcore_axis()
        tile = td.get_tile(node)
        tile_M, tile_N = tile[C_ax_m], tile[C_ax_n]
        tile_K = list(td.get_rstep(node).values())[0]
        condA, condB = True, True
        if A_ax_m < A_ax_k: # MxK
            condA &= tile_K % 32 == 0 and tile_M % 32 == 0
        else: # KxM
            condA &= tile_M % 64 == 0
        if B_ax_n < B_ax_k: # NxK
            condB &= tile_K % 32 == 0 and tile_N % 32 == 0
        else: # KxM
            condB &= tile_N % 64 == 0
        return condA, condB

    def infer_node_smem_usage(self, td: TileDict, node: IRNode):
        value = super().infer_node_smem_usage(td, node)
        if node.get_tag("tensorCoreConfig"):
            use_double_buffer = td.use_cutlass_mma[node] and self.arch.compute_capability >= "80"
            if use_double_buffer: value *= 2
        return value

    def _assign_reduce_step(self, node):
        if not node.get_tag("tensorCoreConfig"):
            return super()._assign_reduce_step(node)
        result = {}
        for k in node.raxis:
            if node.raxis[k] % 32 == 0:
                result[k] = 32
            else:
                return super()._assign_reduce_step(node)
        return result

    def get_node_reduce_step_candidates(self, node):
        if not node.get_tag("tensorCoreConfig"):
            return super().get_node_reduce_step_candidates(node)
        else:
            # must be a a multiple of wmma_k
            return {k : [x * self.wmma_k for x in get_all_factors(node.raxis[k] // self.wmma_k)] for k in node.raxis}

    def check_tile_shape_isvalid(self, td: TileDict):
        for node in self.ordered_nodes:
            if node.get_tag("tensorCoreConfig"):
                ax_m, ax_n = node.get_tag("tensorCoreConfig")
                block_m, block_n = td.tile_map[node][ax_m], td.tile_map[node][ax_n]
                wmma_invalid = [block_m % wmma_m or block_n % wmma_n for wmma_m, wmma_n in [(16, 16), (8, 32), (32, 8)]]
                if all(wmma_invalid):
                    return False
                if any([y % x for x, y in zip(td.tile_map[node], node.get_shape())]):
                    return False
        return super().check_tile_shape_isvalid(td)

    def compute_node_stride_map(self, node: IRNode, td: TileDict):
        if not node.get_tag("tensorCoreConfig"):
            return super().compute_node_stride_map(node, td)
        td.use_cutlass_mma[node] = self._use_cutlass_mma(node, td)
        use_layout = self._can_implement_layout(node, td)
        AS_stride, BS_stride, C_stride = self._compute_tc_strides(node, td.get_tile(node), td.get_rstep(node))
        A_stride, B_stride, _ = self._compute_tc_strides(node, td.get_tile(node))
        output_strides = {int(edge.src_id + len(node.inputs)): C_stride for edge in node.outputs}
        tensor_strides = {}
        # when connected to shared input, should use full stride without rstep
        for i, (stride, stride_full) in enumerate(zip([AS_stride, BS_stride], [A_stride, B_stride])):
            if use_layout[i]: continue
            name = node.reduce_op.input_tensors[i].name
            tensor_strides[name] = stride

            arg_names = [arg.name for arg in node.args]
            if name in arg_names:
                input_id = arg_names.index(name)
                src_node = node.inputs[input_id].src_node
                if not src_node.is_placeholder():
                    tensor_strides[name] = stride_full

        return output_strides, tensor_strides

    def _assign_block_size(self, node: Node, td: TileDict, block_size: int):
        if not node.get_tag("tensorCoreConfig"):
            return super()._assign_block_size(node, td, block_size)
        ax_m, ax_n = node.get_tag("tensorCoreConfig")
        if block_size % self.arch.warp_size != 0:
            return None
        tile, rsteps = td.get_tile(node), td.get_rstep(node)
        warps = block_size // self.arch.warp_size
        ndim = len(tile)
        if td.use_cutlass_mma[node]:
            wmma = self.arch.cutlass_mma
        elif tile[ax_m] > tile[ax_n]:
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
                score += np.prod(shape[edge.dst_id]) / self.arch.bandwidth[1]
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
        codegen_dict.use_tc = self.arch.compute_capability
        codegen_dict.block = tile
        codegen_dict.warp = warp_tile
        codegen_dict.rstep = [int(rsteps[ax]) for ax in node.raxis]
        codegen_dict.wmma = wmma
        codegen_dict.use_cutlass = td.use_cutlass_mma[node]
        codegen_dict.complete_config(node)
        return codegen_dict
