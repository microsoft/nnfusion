import tvm
from tvm import te
from roller.utils import *
import math
from roller.config import *
# from roller.test_config import *


class Op:
    def __init__(self, expr, shape, data_type, use_tc=False) -> None:
        self.expr = expr
        self.shape = shape
        self.use_tc = use_tc
        self.ori_in = []
        self.pad_in = [] 
        self.outs = []
        self.unpad_outs = []
        self.expr_out = self.expr(self.shape, dataType=data_type, for_rtile=False)
        self.input_tensors = self.expr_out[0]
        self.output_tensors =self.expr_out[1]
        self.fused_shape = []
        self.data_type = data_type
        
        for it in self.input_tensors:
            if '_pad' in it.name:
                self.pad_in.append(it)
            else:
                self.ori_in.append(it)

        for ot in self.output_tensors:
            if '_unpad' in ot.name:
                self.unpad_outs.append(ot)
            else:
                self.outs.append(ot)
        # todo: what if multi-output op?
        self.saxis, self.raxis = get_axis_names(self.output_tensors[0])
        if len(self.unpad_outs) > 0:
            self.sche = tvm.te.create_schedule(self.unpad_outs[0].op)
        else:
            self.sche = tvm.te.create_schedule(self.output_tensors[0].op)

        if len(self.expr_out) == 3:
            self.fused_shape_map = self.expr_out[2]
            for a in self.saxis:
                fs = self.fused_shape_map[a]
                dim = 1
                for d in fs:
                    dim *= self.shape[d]
                self.fused_shape.append(dim)
            for a in self.raxis:
                fs = self.fused_shape_map[a]
                dim = 1
                for d in fs:
                    dim *= self.shape[d]
                self.fused_shape.append(dim)

    
        self.spatial_dim = len(self.saxis)
        self._axis_id = {}
        aid = 0
        for axis_name in self.saxis:
            self._axis_id[axis_name] = aid
            aid += 1
        for axis_name in self.raxis:
            self._axis_id[axis_name] = aid
            aid += 1

    def GetUniSchedule(self):
        if not self.use_tc:
            return [1 for _ in range(len(self.SAxis()) + len(self.RAxis()))]
        else:
            return [16 for _ in range(len(self.SAxis()) + len(self.RAxis()))]

    def GetAxisLen(self, axis_name):
        assert axis_name in self._axis_id
        aid = self._axis_id[axis_name]
        return self.Dimensions()[aid]

    def TensorTypeSize(self, tvm_codegen=False):
        tensor_type_size = [[],[]] #input, output
        for t in self.GetInputTensors(tvm_codegen):
            assert isinstance(t, tvm.te.Tensor)
            tensor_type_size[0].append(tvm.runtime.DataType(t.dtype).bits // 8)
        for t in self.GetOutputTensors():
            assert isinstance(t, tvm.te.Tensor)
            tensor_type_size[1].append(tvm.runtime.DataType(t.dtype).bits // 8)
        return tensor_type_size

    def InputTypeSize(self):
        return self.TensorTypeSize()[0][0]

    def OutputTypeSize(self):
        return self.TensorTypeSize()[1][0]

    def TensorDim(self, tvm_codegen=False):
        tensor_dim = [[], []]
        for t in self.GetInputTensors(tvm_codegen):
            assert isinstance(t, tvm.te.Tensor)
            tensor_dim[0].append([int(d) for d in t.shape])
        for t in self.GetOutputTensors():
            assert isinstance(t, tvm.te.Tensor)
            tensor_dim[1].append([int(d) for d in t.shape])
        return tensor_dim

    def ComputeWorkload(self, rtile):
        wk = 1 
        for d in rtile.GetOutputDataTiles()[0]:
            wk *= d
        op_rdim = self.RDimensions()
        tile_rdim = rtile.RDimensions()
        assert len(op_rdim) == len(tile_rdim)
        for i in range(len(op_rdim)):
            aligned_r = math.ceil(op_rdim[i] / tile_rdim[i]) * tile_rdim[i]
            wk *= aligned_r
        tensor_type_size = self.OutputTypeSize()
        return wk * tensor_type_size / 2
        
    def MemWorkload(self, rtile, tile_tensor="output"): #todo
        op_rdim = self.RDimensions()
        tile_sdim = rtile.SDimensions()
        tile_rdim = rtile.RDimensions()
        aligned_op_rdim = []
        assert len(op_rdim) == len(tile_rdim)
        for i in range(len(op_rdim)):
            aligned_r = math.ceil(op_rdim[i] / tile_rdim[i]) * tile_rdim[i]
            aligned_op_rdim.append(aligned_r)

        merge_dim = tile_sdim + aligned_op_rdim
        tmp_rtile = rTile(self.expr, merge_dim, self.SAxis(), self.RAxis(), self.GetTvmOutTensor())

        input_data_tiles = tmp_rtile.GetInputDataTiles()
        output_data_tiles = tmp_rtile.GetOutputDataTiles()

        tensor_type_size = self.TensorTypeSize()
        storage_padding = rtile.GetStoragePadding()

        ret = [[], []] #inputs, outputs

        for i in range(len(input_data_tiles)):
            shape = input_data_tiles[i]
            padding = storage_padding[i] 
            area = 1
            for d in range(len(shape)):
                area *= shape[d] + padding[d]
            ret[0].append(int(area * tensor_type_size[0][i]))

        for i in range(len(output_data_tiles)):
            shape = output_data_tiles[i]
            area = 1
            for d in shape:
                area *= d
            ret[1].append(int(area * tensor_type_size[1][i]))
        return ret
    
    def MemFootprint(self, rtile, tile_tensor="output"):
        input_data_tiles = rtile.GetInputDataTiles()
        tensor_type_size = self.TensorTypeSize()
        storage_padding = rtile.GetStoragePadding()
        inputs_size = [] #inputs
        for i in range(len(input_data_tiles)):
            shape = input_data_tiles[i]
            padding = storage_padding[i] 
            area = 1
            assert len(shape) == len(padding)
            for d in range(len(shape)):
                area *= shape[d] + padding[d]
            inputs_size.append(area * tensor_type_size[0][i])
        
        ret = 0
        for t in inputs_size:
            ret += t
        return ret
    
    def Dimensions(self):
        return self.fused_shape if len(self.fused_shape) > 0 else self.shape
    def SAxis(self):
        return self.saxis
    def RAxis(self):
        return self.raxis
    def SDimensions(self):
        return self.Dimensions()[:len(self.saxis)]
    def RDimensions(self):
        return self.Dimensions()[len(self.saxis):]

    def ReductionAxisLen(self):
        ret = 1
        for rn in self.RDimensions():
            ret *= rn
        return ret
    
    def RegUsage(self, rtile, tile_tensor="output"):
        # reduction axis of reg rtile is 1, which should be defined in rtile
        in_datatile = rtile.GetInputDataTiles()
        out_datatiles = rtile.GetOutputDataTiles()
        ret = 0
        for ins in in_datatile:
            area = 1
            for d in ins:
                area *= d
            ret += area * self.InputTypeSize() / 4
        for outs in out_datatiles:
            area = 1
            for d in outs:
                area *= d
            ret += area * self.InputTypeSize() / 4
        if self.use_tc:
            ret /= 32
        if tile_tensor == "output":
            return ret

    def GetGridSize(self, rtile, tile_tensor="output"):
        if tile_tensor == "output":
            output_data_tile = rtile.GetOutputDataTiles()[0]
            output_tensor_shape = self.GetOutputTensors()[0].shape
            assert len(output_data_tile) == len(output_tensor_shape)
            grid_size = 1
            for i in range(len(output_data_tile)):
                grid_i = int((output_tensor_shape[i] + (output_data_tile[i] - 1)) // output_data_tile[i])
                grid_size *= grid_i
            return grid_size
    
    def GetInputTensors(self, tvm_codegen=False):
        if tvm_codegen and len(self.ori_in) > 0:
            return self.ori_in
        else:
            return self.pad_in if len(self.pad_in) > 0 else self.input_tensors
    
    def GetOutputTensors(self):
        return self.unpad_outs if self.unpad_outs else self.output_tensors

    def IODependent(self): #todo
        slopes = []
        #xs = [0.25, 0.5, 0.75]
        xs = [0.1 * s for s in range(1,10)]
        ys = []
        for x in xs:
            shape = [int(x * d) if int(x * d) > 0 else 1 for d in self.Dimensions()]
            rtile = rTile(self.expr, shape, self.SAxis(), self.RAxis(), self.GetTvmOutTensor())
            compute = 1
            for d in shape:
                compute *= d
            io = sum(self.MemWorkload(rtile)[0])
            y = compute / io
            ys.append(y)
        for i in range(len(xs) - 1):
            dy = ys[i + 1] - ys[i]
            dx = xs[i + 1] - xs[i]  
            s = dy/dx 
            slopes.append(s)
        #print("avg_s=", sum_s/len(slopes))
        # print(min(slopes), sum(slopes)/len(slopes))
        for s in slopes:
            #if abs(s) > 3:
            if min(slopes) > 2 and sum(slopes)/len(slopes) > 4:
                return True
        return False

    def GetTvmOutTensor(self):
        return self.output_tensors[0]