import tvm
from tvm import tir


class PassBase:
    def __init__(self) -> None:
        pass

    def run(self, f: tir.PrimFunc, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.tir.PrimFunc:
        raise NotImplementedError()

    def insert_place(self) -> int:
        return 0

    def get_pass(self) -> tir.transform.PrimFuncPass:
        return self.insert_place(), tir.transform.prim_func_pass(lambda f, mod, ctx: self.run(f, mod, ctx), opt_level=0)
