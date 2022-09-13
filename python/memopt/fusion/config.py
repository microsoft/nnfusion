from typing import Dict, List, Optional
import numpy as np

class TensorCoreExtraConfig:
    def __init__(self, AS_shape, BS_shape, AF_shape, BF_shape, tc_axis) -> None:
        self.AS_shape = AS_shape
        self.BS_shape = BS_shape
        self.AF_shape = AF_shape
        self.BF_shape = BF_shape
        self.tc_axis = tc_axis

class Stride:
    def __init__(self, stride: int = 1, ax: int = -1) -> None:
        # which axis to put stride on
        self._ax = ax
        # the stride size of the axis
        self._stride = stride

    @property
    def ax(self) -> int:
        return self._ax

    @property
    def stride(self) -> int:
        return self._stride

    def compute_strides_from_shape(self, shape: List[int]) -> List[int]:
        ndim = len(shape)
        strides = [1 for _ in shape]
        for i in range(ndim - 2, -1, -1):
            if i == self.ax:
                strides[i] = self.stride
            else:
                strides[i] = strides[i + 1] * shape[i + 1]
        return strides

    def is_valid(self) -> bool:
        return self.ax >= 0

class Config:
    def __init__(self) -> None:
        self.use_tc = False
        # spacial axes tiling info
        self.block = []
        self.thread = []
        # special axes for tensorCore
        self.warp = []
        self.wmma = []
        self.tc_extra_conf: Optional[TensorCoreExtraConfig] = None
        # reduce axes tiling info
        self.rstep = []
        self.reduce_thread = []

        self.block_order = None
        self.output_strides = {}

        # For single node code generation, following args can be ommitted
        # For fusion code generation, following args will be completed by CodeGenerator
        self.shared_inputs = {}
        self.reuse_disabled_inputs = {}

        # Experimental
        self._raxis_order = []
        self._step = []

    def to_dict(self) -> Dict:
        dic = {}
        dic["block"] = self.block
        if self.use_tc:
            dic["warp"] = self.warp
            dic["wmma"] = self.wmma
        else:
            dic["thread"] = self.thread
        dic["rstep"] = self.rstep
        if np.prod(self.reduce_thread) > 1:
            dic["reduce_thread"] = self.reduce_thread
        if self.block_order is not None:
            dic["block_order"] = self.block_order
        if self.use_tc:
            dic["use_tc"] = True
        if self.output_strides:
            dic["strides"] = self.output_strides
        if np.prod(self._step) > 1:
            dic["step"] = self._step
        if self._raxis_order != []:
            dic["raxis_order"] = self._raxis_order
        return dic

    def from_dict(self, dic: Dict) -> "Config":
        self.__init__()
        if "use_tc" in dic:
            self.use_tc = dic["use_tc"]
        self.block = dic["block"]
        if self.use_tc:
            self.warp = dic["warp"]
            self.wmma = dic["wmma"]
        else:
            self.thread = dic["thread"]
        self.rstep = dic["rstep"]
        if "reduce_thread" in dic:
            self.reduce_thread = dic["reduce_thread"]
        else:
            self.reduce_thread = [1 for _ in self.rstep]
        if "block_order" in dic:
            self.block_order = dic["block_order"]
        if "strides" in dic:
            self.output_strides = dic["strides"]
        if "step" in dic:
            self._step = dic["step"]
        if "raxis_order" in dic:
            self._raxis_order = dic["raxis_order"]
        return self

    @property
    def raxis_order(self) -> List[int]:
        if self._raxis_order != []:
            return self._raxis_order
        return list(range(len(self.rstep)))

    @property
    def step(self) -> List[int]:
        if self._step != []:
            return self._step
        return [1 for _ in self.block]

    def __repr__(self) -> str:
        return str(self.to_dict())

    def complete_config(self, node, C_ax_m: int, C_ax_n: int):
        if not self.use_tc:
            return self
        wmma_m, wmma_n, wmma_k = self.wmma
        CL_shape = [1 for _ in node.saxis]
        CL_shape[C_ax_m] = wmma_m
        CL_shape[C_ax_n] = wmma_n

        shapes = node.infer_dependency_reduce_inputs(CL_shape, {x : 1 for x in node.raxis})
        A_deps, B_deps = shapes.values()
        A_ax_m = A_deps.index(wmma_m)
        B_ax_n = B_deps.index(wmma_n)
        shapes = node.infer_dependency_reduce_inputs([1 for _ in node.saxis], {x : wmma_k for x in node.raxis})
        A_deps, B_deps = shapes.values()
        A_ax_k = A_deps.index(wmma_k)
        B_ax_k = B_deps.index(wmma_k)
        tc_axis = [A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n]

        shapes = node.infer_dependency_reduce_inputs(self.block, {x : self.rstep[0] for x in node.raxis})
        AS_shape, BS_shape = shapes.values()

        shapes = node.infer_dependency_reduce_inputs(self.warp, {x : wmma_k for x in node.raxis})
        AF_shape, BF_shape = shapes.values()

        self.tc_extra_conf = TensorCoreExtraConfig(AS_shape, BS_shape, AF_shape, BF_shape, tc_axis)
        return self
