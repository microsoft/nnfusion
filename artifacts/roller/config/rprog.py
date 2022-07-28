import math
import tvm
from roller.utils import *

def num_tiles(large_shape, small_shape):
    ret = 1
    for d1, d2 in zip(large_shape, small_shape):
        ret *= math.ceil(d1 / d2)
    return ret


class rProg:
    def __init__(self, num_level, op) -> None:
        self.rTiles = {}
        self.num_level = num_level

        self.op = op
        self.full_shape = self.op.Dimensions()
        self.expr = self.op.expr
        self.sche = self.op.sche
        self.saxis = self.op.SAxis()
        self.raxis = self.op.RAxis()

        self.spatial_dim = len(self.saxis)
        self._axis_id = {}
        aid = 0
        for axis_name in self.saxis:
            self._axis_id[axis_name] = aid
            aid += 1
        for axis_name in self.raxis:
            self._axis_id[axis_name] = aid
            aid += 1

    def AddTile(self, mem_level, rtile):
        self.rTiles[mem_level] = rtile

    def DeleteTile(self, mem_level):
        del self.rTiles[mem_level]

    def GetTile(self, mem_level):
        return self.rTiles[mem_level]

    def UpdateTile(self, new_rtile, mem_level):
        assert mem_level in self.rTiles
        self.rTiles[mem_level] = new_rtile

    def GetAxisConfig(self, axis_name):
        assert axis_name in self._axis_id
        aid = self._axis_id[axis_name]
        ret = []
        for l in range(self.num_level + 1):
            ret.append(self.rTiles[l].shape[aid])
        return ret

    def GetParallelism(self, mem_level):
        if mem_level == 0:
            return num_tiles(self.full_shape[:self.spatial_dim],
                             self.rTiles[0].SDimensions())
        else:
            return num_tiles(self.rTiles[mem_level - 1].SDimensions(),
                             self.rTiles[mem_level].SDimensions())

    def Dump(self):
        ret = ""
        for i in range(self.num_level + 1):
            if i in self.rTiles:
                rtile_str = self.GetTile(i).Dump()
                level_str = "[level {}: {}]".format(i, rtile_str)
                ret += level_str
        return ret

    def GetTVMSchedule(self):
        assert isinstance(self.sche, tvm.te.Schedule)
        return self.sche

    def Expression(self):
        return self.expr

    def Dimensions(self):
        return self.full_shape

    def copy(self):
        newProg = rProg(self.num_level, self.op)
        for mem_level in self.rTiles:
            newProg.AddTile(mem_level, self.rTiles[mem_level].copy())
        return newProg
