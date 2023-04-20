from ast_analyzer.grad import annotations as anno
import gast

class Namer:
    def __init__(self, prefix="__tmp"):
        self.start_id = 0
        self.prefix = prefix
    
    def next(self):
        ret = f"{self.prefix}_{self.start_id}"
        self.start_id += 1
        return ret


def copy_anno(dst, src, names):
    for name in names:
        if anno.hasanno(src, name):
            anno.setanno(dst, name, anno.getanno(src, name))


def is_call_stmt(stmt):
    return isinstance(stmt, gast.Assign) and isinstance(stmt.value, gast.Call)
