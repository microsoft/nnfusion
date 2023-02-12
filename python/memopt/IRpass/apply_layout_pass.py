from typing import Callable, Dict, Union

import tvm
from tvm import tir

from .pass_base import PassBase


class ApplyLayoutPass(PassBase):
    def __init__(self, layout_map: Dict[str, Callable]) -> None:
        super().__init__()
        self.layout_map = layout_map

    def run(self, f: tir.PrimFunc, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.tir.PrimFunc:
        def process(op : Union[tvm.tir.BufferStore, tvm.tir.BufferLoad]):
            if not op.buffer.name in self.layout_map:
                return op
            layout = self.layout_map[op.buffer.name]

            assert(len(op.indices) == 1) # already flattened

            new_indices = [layout(op.indices[0])]

            if isinstance(op, tvm.tir.BufferLoad):
                return tvm.tir.BufferLoad(op.buffer, new_indices, op.span)
            else:
                return tvm.tir.BufferStore(op.buffer, op.value, new_indices, op.span)

        new_body = tvm.tir.stmt_functor.ir_transform(f.body, None, process, ["tir.BufferStore", "tir.BufferLoad"])
        return f.with_body(new_body)

    def insert_place(self) -> int:
        return 1
