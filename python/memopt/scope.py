from typing import Dict
import tvm

_current_scope = None

class Scope(Dict):
    def __init__(self, schedule):
        self.schedule = schedule
        self.bounds = tvm.te.schedule.InferBound(self.schedule)
        self.shared_mem_outputs = []

    def __enter__(self):
        global _current_scope
        _current_scope = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _current_scope
        _current_scope = None

def get_scope():
    return _current_scope
