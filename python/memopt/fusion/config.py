from typing import Dict, List
import numpy as np

class Config:
    def __init__(self) -> None:
        self.use_tc = False
        # spacial axes tiling info
        self.block = []
        self.thread = []
        # special axes for tensorCore
        self.warp = []
        self.wmma = []
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
