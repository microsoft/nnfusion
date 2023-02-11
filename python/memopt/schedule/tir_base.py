import tvm
from tvm import te, tir

from .scheduler_base import SchedulerBase


class TESchedulerBase(SchedulerBase):
    def create_tir_schedule(self) -> tir.Schedule:
        workload = te.create_prim_func(self.args)
        ir_module = tvm.IRModule({"main": workload})
        return tir.Schedule(ir_module)
