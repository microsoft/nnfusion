import gast as ast
import onnx
from onnx import helper, numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto, ValueInfoProto, OperatorSetIdProto
import astunparse
import torch
import numpy as np
import copy
from contextlib import contextmanager
from ast_analyzer.grad import annotations as anno
from ast_analyzer.shape_inference.types import *

from .utils import *
from .export_functions import pytorch_func_export, pytorch_layer_export, builtin_func_export, general_func_export
from .pytorch_layers import pytorch_layer_initializer
from .node import OnnxNodes
from .for_parser import get_for_parser
from ast_analyzer.grad import annotations as anno


def rewrite_graph_udfs(graph, to_add, op_names):
    assert(isinstance(graph, onnx.GraphProto))
    for node in graph.node:
        if node.op_type in op_names:
            node.input.extend(to_add)
        for attr in node.attribute:
            if attr.HasField('g'):
                assert(isinstance(attr.g, onnx.GraphProto))
                rewrite_graph_udfs(attr.g, to_add, op_names)

class ExportEngine(ast.NodeVisitor):
    def __init__(self, name, ast_info={}, fix_input=False, attrs_order=[], func2name={}):
        # self.type_dict = type_dict # type: Map[ASTNode, Type] astnode -> type
        self.ast_info = ast_info # type: Map[ASTNode, str] astnode -> name of live variables
        self.name_dict = {} # type: Map[str, str] name -> name_in_onnx
        self.name_cache = {} # type: Map[str, int] name -> last_id to ensure SSA in onnx
        self.output_name = {} # type: Map[ASTNode, str] astnode -> name assigned by parent nodes
        self.output_nodes = None # type: Optional[?]
        self.initializers = {} # type: Map[obj(Model params), List[TensorProto]] obj -> onnxnode
        self.model_name = name # type: str
        self.value_info = {} # type: Map[str, ValueInfoProto] name_in_onnx -> value_info
        self.model_params_info = {} # type: Map[str, Map[str, ValueInfoProto]]
        self.arg_name = {} # type: Map[str, (int, type)] arg_name -> order in arg list
        self.input_nodes = {} # type: Map[str, ValueInfoProto] input arg name -> value_info for graph inputs
        self.onnx2py = {} # type: Map[str, (ASTNode, Type)] name_in_onnx -> python code to extract it + type of the node
        self.write_through = {} # type: Map[str, ValueInfoProto] name of variables to copy in new_graph_space
        self.fix_input = fix_input # type: bool the onnx graph must the same input as the function call
        self.arg_value_infos = {} # type: Map[str, ValueInfoProto] name in onnxgraph -> valueinfo for args
        self.attrs_order = attrs_order # type: List[str] the order of self.xxx
        self.func2name = func2name # type: Map[callable, str]: func_inst to the recursive node name

    def gen_name(self, name=None):  # str -> str
        if isinstance(name, str):
            pass
        elif name is None:
            name = "@tmp"
        else:
            raise NotImplementedError(str(type(name)))

        if name not in self.name_cache:
            self.name_cache[name] = 0
        else:
            self.name_cache[name] += 1
        return name + "_{}".format(self.name_cache[name])

    def get_or_gen_name(self, node):  # ASTNode -> str
        if node in self.output_name:
            return self.output_name[node]
        else:
            return self.gen_name()

    def get_type_of_node(self, node):  # ASTNode -> Type
        return anno.getanno(node, 'type')

    def set_output_name(self, node, name):  # ASTNode, str -> None
        self.output_name[node] = self.gen_name(name)

    def add_initializer(self, obj, tensors, value_infos, name): # name: self_xxx means self.xxx
        # type: obj, List[TensorProto], Map[str, ValueInfoProto] -> None
        # assert obj not in self.initializers
        if obj not in self.initializers:
            assert not isinstance(obj, str)  # make sure not passing a "name"
            assert isinstance(tensors, list)
            self.initializers[obj] = tensors
            self.model_params_info[name] = value_infos

    def execute_list(self, stmts, args, rets, check_model=True, wrap_recursion=False, is_training=True):
        for i, (arg_name, arg_type) in enumerate(args):
            self.arg_name[arg_name] = (i + 1, arg_type)  # the first arg is "self"
        compute_nodes = []
        value_infos = self.arg_value_infos
        for stmt in stmts:
            onnx_nodes = self.visit(stmt)
            compute_nodes += onnx_nodes.def_nodes
            value_infos.update(onnx_nodes.def_value_infos)
        output_nodes = [value_infos[self.name_dict[x]] for x in rets]
        input_nodes = []
        for node_id, _ in args:
            # None will not be used in the code, so it can only be added here
            is_onnx_input = True
            if node_id not in self.input_nodes:
                is_onnx_input = self.try_add_argument(node_id)
            if is_onnx_input:
                input_nodes.append(self.input_nodes[node_id])
        output_node_names = [x.name for x in output_nodes]
        input_node_names = [x.name for x in self.input_nodes.values()]
        for i in range(len(output_nodes)):
            if (output_nodes[i].name in output_node_names[:i] or output_nodes[i].name in input_node_names):
                new_name = self.gen_name(output_nodes[i].name)
                compute_nodes.append(helper.make_node("Identity", [output_nodes[i].name], [new_name]))
                output_nodes[i] = copy.deepcopy(output_nodes[i])
                output_nodes[i].name = new_name

        graph_def = helper.make_graph(
            compute_nodes,
            self.model_name,
            input_nodes,
            output_nodes,
        )

        udf_inputs = []
        # to do e2e test. ort.get_inputs ignores optional input

        if is_training:
            for attr in self.attrs_order:
                encoded = "self_" + attr
                if encoded in self.model_params_info:
                    infos = self.model_params_info[encoded]
                    for info in infos.values():
                        assert(isinstance(info, ValueInfoProto))
                        graph_def.input.append(info)
                        udf_inputs.append(info.name)
            
            for name, infos in self.model_params_info.items():
                if not name.startswith("self_") or not name[5:] in self.attrs_order:
                    for info in infos.values():
                        assert(isinstance(info, ValueInfoProto))
                        graph_def.input.append(info)
                        udf_inputs.append(info.name)

        if len(self.func2name) > 1:
            raise NotImplementedError
        graph_def.value_info.extend(value_infos.values())
        if len(self.func2name) > 0:
            rewrite_graph_udfs(graph_def, udf_inputs, self.func2name.values())
            if wrap_recursion:
                assert(len(self.func2name) == 1)
                graph_name = list(self.func2name.values())[0]
                graph_def.name = graph_name
                recursion_node = onnx.helper.make_node('Recursion', inputs=[x.name for x in graph_def.input], outputs = [x.name for x in graph_def.output], body=graph_def)
                graph_def_old = graph_def
                graph_def = onnx.helper.make_graph(
                    nodes=[recursion_node],
                    name=self.gen_name(self.model_name),
                    inputs=copy.deepcopy(graph_def_old.input),
                    outputs=copy.deepcopy(graph_def_old.output)
                )

        if not is_training:
            for inis in self.initializers.values():
                graph_def.initializer.extend(inis)

        # for info in graph_def.value_info:
        #     print("[shape]", info.name + ":", str([x.dim_value for x in info.type.tensor_type.shape.dim]))

        # print("[graph meta]")
        # print("inputs", [x.name for x in graph_def.input])
        # print("outputs", [x.name for x in output_nodes])
        # print("name_dict", self.name_dict)
        # print(self.onnx2py)

        self.result_args = []
        self.result_node_to_order = {}
        self.result_arg_type = []
        for i, x in enumerate(graph_def.input):
            node, ty = self.onnx2py[x.name]
            self.result_args.append(node)
            self.result_arg_type.append(ty)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'self' and isinstance(node.ctx, ast.Load):
                self.result_node_to_order[node.attr] = i

        self.result_num_input = len(graph_def.input)
        self.result_num_output = len(graph_def.output)

        # if self.result_num_input == 0:
        #     print("[GenOnnx] error: no input") # TODO nnfusion does not support it
        #     return None

        # graph_def = self.visit(node.body[0])
        opset = OperatorSetIdProto()
        opset.version = 11
        model = helper.make_model(graph_def, producer_name='autogen', opset_imports=[opset])
        print("[generated graph]")
        print(onnx.helper.printable_graph(model.graph))
        if check_model:
            onnx.checker.check_model(model)
        return model

    def execute(self, node):  # ast.Module
        assert isinstance(node, ast.Module)
        return self.visit(node)

    def register_value_info(self, value_info):  # ValueInfoProto -> None
        assert(value_info.name not in self.value_info)
        self.value_info[value_info.name] = value_info

    def visit_Module(self, node):  # ASTNode -> ModelProto
        assert len(node.body) == 1
        assert isinstance(node.body[0], ast.FunctionDef)
        graph_def = self.visit(node.body[0])
        opset = OperatorSetIdProto()
        opset.version = 11
        model = helper.make_model(graph_def, producer_name='autogen', opset_imports=[opset])
        # print("[generated graph]")
        # print(onnx.helper.printable_graph(model.graph))
        onnx.checker.check_model(model)
        return model

    def visit_FunctionDef(self, node):  # ASTNode -> GraphProto
        assert node.args.args[0].id == 'self'
        arg_tys = self.get_type_of_node(node).argty

        for i, arg_node in enumerate(node.args.args):
            if i == 0:
                continue  # the first arg is "self"
            self.arg_name[arg_node.id] = (i, anno.getanno(arg_node, 'type'))

        compute_nodes = []

        if self.fix_input:
            for arg in node.args.args[1:]:
                assert(self.try_add_argument(arg.id))
        
        value_infos = self.arg_value_infos
        
        for stmt in node.body:
            onnx_nodes = self.visit(stmt)
            compute_nodes += onnx_nodes.def_nodes
            value_infos.update(onnx_nodes.def_value_infos)
        assert self.output_nodes is not None

        output_nodes = [value_infos[x] for x in self.output_nodes]

        input_nodes = []
        for arg in node.args.args[1:]:
            node_id = arg.id
            # None will not be used in the code, so it can only be added here
            is_onnx_input = True
            if node_id not in self.input_nodes:
                is_onnx_input = self.try_add_argument(node_id)
            if is_onnx_input:
                input_nodes.append(self.input_nodes[node_id])

        graph_def = helper.make_graph(
            compute_nodes,
            self.model_name,
            input_nodes,
            output_nodes,
        )

        # to do e2e test. ort.get_inputs ignores optional input
        # for inis in self.initializers.values():
        #     graph_def.initializer.extend(inis)
        for attr in self.attrs_order:
            encoded = "self_" + attr
            if encoded in self.model_params_info:
                infos = self.model_params_info[encoded]
                for info in infos.values():
                    assert(isinstance(info, ValueInfoProto))
                    graph_def.input.append(info)
        
        for name, infos in self.model_params_info.items():
            if not name.startswith("self_") or not name[5:] in self.attrs_order:
                for info in infos.values():
                    assert(isinstance(info, ValueInfoProto))
                    graph_def.input.append(info)

        graph_def.value_info.extend(value_infos.values())

        # print("[graph meta]")
        # print("inputs", [x.name for x in graph_def.input])
        # print("outputs", [x.name for x in output_nodes])
        # print("name_dict", self.name_dict)
        # print(self.onnx2py)

        self.result_pre_onnx_nodes = []
        self.result_node_to_order = {}
        self.result_arg_type = []
        for i, x in enumerate(graph_def.input):
            node, ty = self.onnx2py[x.name]
            self.result_pre_onnx_nodes.append(
                ast.Assign(
                    targets=[ast.Name(id='_tensor_{}'.format(
                        i), ctx=ast.Store(), annotation=None, type_comment=None)],
                    value=node
                )
            )
            self.result_arg_type.append(ty)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'self' and isinstance(node.ctx, ast.Load):
                self.result_node_to_order[node.attr] = i

        self.result_num_input = len(graph_def.input)
        self.result_num_output = len(graph_def.output)

        return graph_def

    def visit_Assign(self, node):  # ASTNode -> OnnxNodes
        assert len(node.targets) == 1
        target = node.targets[0]
        if isinstance(target, ast.Name):
            self.set_output_name(node.value, target.id)
            val_nodes = self.visit(node.value)
            self.name_dict[target.id] = val_nodes.out_node[0]
            return val_nodes
        if isinstance(target, ast.Subscript):
            if isinstance(target.value, ast.Name):
                target_value_node = self.visit(target.value)
                if isinstance(target.slice, ast.Index) and isinstance(target.slice.value, ast.Name): # a[b] = xxx
                    slice_type = self.get_type_of_node(target.slice.value)
                    if (isinstance(slice_type, TyNum) and slice_type.is_int()) or \
                       (isinstance(slice_type, TyTensor) and slice_type.is_int() and slice_type.is_scalar()):
                        # print("[op] ScatterND", astunparse.unparse(target))
                        target_id = target.value.id
                        self.set_output_name(node.value, target_id)
                        val_nodes = self.visit(node.value)
                        assert(len(val_nodes.out_node) == 1)
                        index_nodes = self.visit(target.slice.value)
                        assert(len(index_nodes.out_node) == 1)

                        index_shape_name = self.gen_name()
                        index_shape = helper.make_node(
                            'Constant',
                            [],
                            [index_shape_name],
                            value=numpy_helper.from_array(np.full((1,), 1))
                        )
                        index_shape_node = OnnxNodes(
                            def_nodes=[index_shape], out_node = [index_shape_name], def_value_infos={}
                        )
                        index_name = self.get_or_gen_name(target)
                        index_nodes.set_output(
                            helper.make_node(
                                'Reshape',
                                [index_nodes.out_node[0], index_shape_name],
                                [index_name],
                            ),
                            index_name
                        )
                        index_shape_node += index_nodes
                        index_shape_node += val_nodes
                        out_name = self.get_or_gen_name(target.slice.value.id)
                        onnx_node = helper.make_node(
                            'ScatterND',
                            [self.name_dict[target.value.id], index_name, val_nodes.out_node[0]],
                            [out_name]
                        )
                        value_info = type_to_value_info(out_name, self.get_type_of_node(node.targets[0].value), self)
                        index_shape_node.set_output(onnx_node, out_name, value_info)
                        self.name_dict[target.value.id] = out_name
                        return index_shape_node
            
                if isinstance(target.slice, ast.Index) and isinstance(target.slice.value, ast.Constant): # a[2] = xxx
                    val_nodes = self.visit(node.value)
                    assert(len(val_nodes.out_node) == 1)
                    slice_type = self.get_type_of_node(target.slice.value)
                    index_name = self.gen_name()
                    index_node = helper.make_node(
                        'Constant',
                        [],
                        [index_name],
                        value=numpy_helper.from_array(np.full((1,), target.slice.value.value))
                    )
                    index_onnx_node = OnnxNodes(
                        def_nodes=[index_node], out_node=[index_name], def_value_infos={}
                    )
                    index_onnx_node += val_nodes
                    out_name = self.gen_name()
                    onnx_node = helper.make_node(
                        'ScatterND',
                        [self.name_dict[target.value.id], index_name, val_nodes.out_node[0]],
                        [out_name]
                    )
                    value_info = type_to_value_info(out_name, self.get_type_of_node(node.targets[0].value), self)
                    index_onnx_node.set_output(onnx_node, out_name, value_info)
                    self.name_dict[target.value.id] = out_name
                    return index_onnx_node

                if isinstance(target.slice, ast.ExtSlice): # a[:, :, x,] = xxx only support the indexing of one dimension and the index must be a scalar now
                    val_nodes = self.visit(node.value)
                    assert(len(val_nodes.out_node) == 1)
                    dim_id = -1
                    for i, dim in enumerate(target.slice.dims):
                        if isinstance(dim, ast.Index) and isinstance(dim.value, (ast.Name, ast.Constant)):
                            assert dim_id == -1, "only support the indexing of one dimension"
                            dim_id = i
                        elif isinstance(dim, ast.Slice):
                            if dim.lower is not None or dim.upper is not None or dim.step is not None:
                                raise NotImplementedError
                        else:
                            raise ValueError("invalid slice type")
                    index_node = self.visit(target.slice.dims[dim_id].value)
                    assert(len(index_node.out_node) == 1)

                    index_type = self.get_type_of_node(target.slice.dims[dim_id].value)
                    target_shape = self.get_type_of_node(target.value).unwrapped_shape()
                    index_shape_prefix = list(target_shape[:dim_id])
                    index_base_tensor_list = []
                    mask_list = []
                    for i in range(dim_id):
                        dim_size = target_shape[i]
                        tmp_shape = [1] * len(index_shape_prefix)
                        tmp_shape[i] = dim_size
                        val = np.arange(dim_size).reshape(tmp_shape)
                        val = np.broadcast_to(val, index_shape_prefix)
                        index_base_tensor_list.append(val)
                        mask_list.append(np.ones_like(val, dtype=np.bool))
                    if index_type.value is not None:
                        index_base_tensor_list.append(np.full(index_shape_prefix, index_type.value, dtype=np.int64))
                    else:
                        index_base_tensor_list.append(np.zeros(index_shape_prefix, dtype=np.int64))
                    index_base_tensor = np.stack(index_base_tensor_list, axis=dim_id)
                    if index_type.value is not None:
                        index_tensor_name = self.gen_name()
                        index_tensor_node = helper.make_node(
                            'Constant',
                            [],
                            [index_tensor_name],
                            value=numpy_helper.from_array(index_base_tensor)
                        )
                        out_name = self.gen_name()
                        onnx_node = helper.make_node(
                            'ScatterND',
                            [self.name_dict[target.value.id], index_tensor_name, val_nodes.out_node[0]],
                            [out_name]
                        )

                        value_info = type_to_value_info(out_name, self.get_type_of_node(node.targets[0].value), self)
                        self.name_dict[target.value.id] = out_name
                        
                        return OnnxNodes(
                            def_nodes=val_nodes.def_nodes + index_node.def_nodes + [index_tensor_node, onnx_node],
                            out_node=[out_name],
                            def_value_infos={out_name: value_info, **val_nodes.def_value_infos, **index_node.def_value_infos}
                        )
                        
                    else:
                        mask_list.append(np.zeros(index_shape_prefix, dtype=np.bool))
                        mask = np.stack(mask_list, axis=dim_id)

                        index_base_name = self.gen_name()
                        index_base_node = helper.make_node(
                            'Constant',
                            [],
                            [index_base_name],
                            value=numpy_helper.from_array(index_base_tensor)
                        )
                        
                        mask_name = self.gen_name()
                        mask_node = helper.make_node(
                            'Constant',
                            [],
                            [mask_name],
                            value=numpy_helper.from_array(mask)
                        )

                        index_tensor_name = self.gen_name()
                        index_tensor_node = helper.make_node(
                            'Where',
                            [mask_name, index_base_name, index_node.out_node[0]],
                            [index_tensor_name]
                        )

                        out_name = self.gen_name()
                        onnx_node = helper.make_node(
                            'ScatterND',
                            [self.name_dict[target.value.id], index_tensor_name, val_nodes.out_node[0]],
                            [out_name]
                        )
                        value_info = type_to_value_info(out_name, self.get_type_of_node(node.targets[0].value), self)
                        self.name_dict[target.value.id] = out_name
                        
                        return OnnxNodes(
                            def_nodes=val_nodes.def_nodes + index_node.def_nodes + [index_base_node, mask_node, index_tensor_node, onnx_node],
                            out_node=[out_name],
                            def_value_infos={out_name: value_info, **val_nodes.def_value_infos, **index_node.def_value_infos}
                        )

        if isinstance(target, ast.Tuple):
            for target_node in target.elts:
                assert(isinstance(target_node, ast.Name))
            val_nodes = self.visit(node.value)
            assert(len(val_nodes.out_node) == len(target.elts))
            for ast_node, onnx_node in zip(target.elts, val_nodes.out_node):
                self.name_dict[ast_node.id] = onnx_node
            return val_nodes
        print(astunparse.unparse(node))
        raise NotImplementedError

    def visit_Return(self, node):  # ASTNode -> OnnxNodes
        if node.value is None:
            raise NotImplementedError
        if self.output_nodes is not None:
            # only allow one output node in each model
            raise NotImplementedError
        val_nodes = self.visit(node.value)
        self.output_nodes = val_nodes.out_node

        return OnnxNodes(
            def_nodes=val_nodes.def_nodes,
            def_value_infos=val_nodes.def_value_infos,
            out_node=None
        )

    def visit_Call(self, node):  # ASTNode -> OnnxNodes
        func = node._func_inst

        if func in pytorch_func_export:
            return pytorch_func_export[func](node, func, self)
        elif type(func) in pytorch_layer_export:
            return pytorch_layer_export[type(func)](node, func,  self)
        elif func in builtin_func_export:
            return builtin_func_export[func](node, func, self)
        elif func in self.func2name:
            print("[to_onnx.engine] match name", self.func2name[func])
            func_onnx_nodes = general_func_export(node, func, self, self.func2name[func])
            return func_onnx_nodes
        raise NotImplementedError(
            "to_onnx.engine: not supported {}, {}".format(func, astunparse.unparse(node)))

    def export_layer(self, node, func_inst):
        if isinstance(node, ast.Attribute):
            name = self.gen_name(node.attr)
        else:
            name = self.gen_name()

        if type(func_inst) in pytorch_layer_initializer:
            initializers, value_infos, extractor, types = pytorch_layer_initializer[type(func_inst)](
                func_inst, name, node, self)
            for v, e, t in zip(value_infos.values(), extractor, types):
                self.onnx2py[v.name] = (e, t)
            self.add_initializer(func_inst, initializers, value_infos, "{}_{}".format(
                node.value.id if isinstance(node.value, ast.Name) else self.gen_name(),
                node.attr
            ) if isinstance(node, ast.Attribute) else self.gen_name())
            return OnnxNodes(out_node=[ini.name for ini in self.initializers[func_inst]], def_value_infos=value_infos)
        else:
            print("node", astunparse.unparse(node))
            raise NotImplementedError()

    def visit_Attribute(self, node):  # ASTNode -> OnnxNodes
        # ty_obj = self.get_type_of_node(node)
        ty_value = self.get_type_of_node(node.value)
        if isinstance(ty_value, TyUserDefinedClass):  # TODO: is ast.load
            x = getattr(ty_value.instance, node.attr)
            if x in self.initializers:
                return OnnxNodes(out_node=[ini.name for ini in self.initializers[x]])
            else:
                func_inst = getattr(ty_value.instance, node.attr)
                return self.export_layer(node, func_inst)
        else:
            raise NotImplementedError

    def visit_BinOp(self, node):  # ASTNode -> OnnxNodes
        node_l = self.visit(node.left)
        node_r = self.visit(node.right)
        ty_l = self.get_type_of_node(node.left)
        ty_r = self.get_type_of_node(node.right)

        valid_l = (isinstance(ty_l, TyTensor) and ty_l.is_fixed_shape()) or isinstance(ty_l, TyNum)
        valid_r = (isinstance(ty_r, TyTensor) and ty_r.is_fixed_shape()) or isinstance(ty_r, TyNum)
       
        if not valid_l or not valid_r:
            print(astunparse.unparse(node))
            raise NotImplementedError("visit_BinOp", ty_l, ty_r)

        if isinstance(node.op, ast.FloorDiv):
            assert(isinstance(ty_l, TyNum) and ty_l.kind == 1)
            assert(isinstance(ty_r, TyNum) and ty_r.kind == 1)

        ast2onnx = {
            ast.Add: 'Add',
            ast.Sub: 'Sub',
            ast.Mult: 'Mul',
            ast.Div: 'Div',
            ast.Pow: 'Pow',
            ast.Mod: 'Mod',
            ast.FloorDiv: 'Div',
            ast.BitOr: 'Or',
            ast.BitAnd: 'And',
        }
        name = self.get_or_gen_name(node)
        assert(len(node_l.out_node) == 1)
        assert(len(node_r.out_node) == 1)
        onnx_node = onnx.helper.make_node(
            ast2onnx[type(node.op)],
            inputs=[node_l.out_node[0], node_r.out_node[0]],
            outputs=[name]
        )
        value_info = type_to_value_info(name, self.get_type_of_node(node), self)
        ret_nodes = node_l + node_r
        ret_nodes.set_output(onnx_node, name, value_info)
        return ret_nodes

    def try_add_argument(self, node_id):
        if node_id not in self.name_dict and node_id in self.arg_name:
            # is an argument
            arg_name = self.gen_name(node_id)
            _, node_type = self.arg_name[node_id]
            if isinstance(node_type, TyNone):
                return False
            self.name_dict[node_id] = arg_name
            arg_node = type_to_value_info(
                arg_name, node_type, self)
            self.input_nodes[node_id] = arg_node
            self.onnx2py[arg_name] = (ast.Name(id=copy.deepcopy(node_id), ctx=ast.Load(), annotation=None, type_comment=None), node_type)
            self.write_through[node_id] = arg_node
            self.arg_value_infos[arg_name] = arg_node
            return True
        return False

    def visit_Name(self, node):  # ASTNode -> OnnxNodes
        if isinstance(node.ctx, ast.Load):
            self.try_add_argument(node.id)
            return OnnxNodes(out_node=[self.name_dict[node.id]])
        else:
            raise NotImplementedError

    def proc_slice(self, node):  # ASTNode -> OnnxNodes, int(axis)
        if isinstance(node, ast.Index):
            return self.visit(node.value), 0
        elif isinstance(node, ast.ExtSlice):
            axis = -1
            for i, dim in enumerate(node.dims):
                if isinstance(dim, ast.Slice):
                    if dim.lower is not None or dim.upper is not None or dim.step is not None:
                        raise NotImplementedError
                elif isinstance(dim, ast.Index):
                    if axis != -1: raise NotImplementedError
                    axis = i
                    onnx_node = self.visit(dim.value)
                else:
                    raise NotImplementedError
            return onnx_node, axis
        else:
            raise NotImplementedError

    def visit_Subscript(self, node):  # ASTNode -> OnnxNodes
        assert(isinstance(node.ctx, ast.Load))  # will not support index assign
        ty_value = self.get_type_of_node(node.value)
        onnx_value = self.visit(node.value)
        onnx_slice, axis = self.proc_slice(node.slice)
        if isinstance(ty_value, TyTensor) or onnx_value.out_is_tensor():
            assert(len(onnx_value.out_node) == 1)
            assert(len(onnx_slice.out_node) == 1)
            name = self.get_or_gen_name(node)
            # print("[op] gather", astunparse.unparse(node))
            onnx_node = onnx.helper.make_node(
                'Gather',
                inputs=[onnx_value.out_node[0], onnx_slice.out_node[0]],
                outputs=[name],
                axis=axis
            )
            ret_nodes = onnx_value + onnx_slice
            value_info = type_to_value_info(name, self.get_type_of_node(node), self)
            ret_nodes.set_output(onnx_node, name, value_info)
            return ret_nodes
        else:
            raise NotImplementedError

    def visit_Constant(self, node):  # ASTNode -> OnnxNodes
        name = self.get_or_gen_name(node)
        ty = self.get_type_of_node(node)
        if isinstance(ty, TyNum):
            # np.full can select the right type for int and float
            if isinstance(ty.value, bool):
                dtype = np.bool_
            elif isinstance(ty.value, int):
                dtype = np.int64
            elif isinstance(ty.value, float):
                dtype = np.float32
            else:
                raise NotImplementedError
            onnx_node = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=[name],
                value=numpy_helper.from_array(np.full((), ty.value, dtype=dtype))
            )
            value_info = type_to_value_info(name, ty, self)
            return OnnxNodes(
                def_nodes=[onnx_node],
                def_value_infos={name: value_info},
                out_node=[name]
            )
        
        raise NotImplementedError

    @contextmanager
    def new_graph_space(self):
        name_dict_old = self.name_dict.copy()
        yield
        name_dict_cur = self.name_dict.copy()
        self.name_dict = name_dict_old
        for name, valueInfo in self.write_through.items():
            if name in name_dict_cur and name not in self.name_dict:
                self.name_dict[name] = valueInfo.name
                # self.value_info[self.name_dict[name]] = valueInfo
        # TODO: nested control flow for write_through
        # self.write_through.clear()

    def visit_For(self, node):  # ASTNode -> OnnxNodes
        for_parser = get_for_parser(node.target, node.iter)
        trip_count = for_parser.trip_count(self)
        ret_node = trip_count
        if len(node.orelse) > 0:
            raise NotImplementedError
        cond_in_name = self.gen_name('cond_in')
        cond_in = onnx.helper.make_tensor_value_info(cond_in_name, onnx.TensorProto.BOOL, [])
        cond_in_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[cond_in_name],
            value=onnx.helper.make_tensor(
                name=self.gen_name(),
                data_type=onnx.TensorProto.BOOL,
                dims=[],
                vals=[1], # numpy does not support zero-dimention tensor
            )
        )
        ret_node.def_nodes.append(cond_in_node)

        with self.new_graph_space():
            iter_count = scaler_to_value_info(
                self.get_or_gen_name('_iter'),
                onnx.TensorProto.INT64,
                self
            )
            target_onodes = for_parser.get_target(iter_count, self)
            live_names = self.ast_info[node]
            graph_value_info_in = [iter_count, cond_in]

            for live_name in live_names:
                name_in = self.gen_name(live_name)
                self.try_add_argument(live_name)
                value_info = value_info_with_name(
                    self.value_info[self.name_dict[live_name]], name_in, self)
                self.name_dict[live_name] = name_in
                graph_value_info_in.append(value_info)

            compute_nodes = target_onodes.def_nodes
            value_infos = target_onodes.def_value_infos
            for stmt in node.body:
                onnx_nodes = self.visit(stmt)
                compute_nodes += onnx_nodes.def_nodes
                value_infos.update(onnx_nodes.def_value_infos)

            graph_value_info_out = [cond_in]
            for live_name in live_names:
                graph_value_info_out.append(
                    self.value_info[self.name_dict[live_name]])

            body_graph = onnx.helper.make_graph(
                compute_nodes,
                self.gen_name('FOR_LOOP'),
                graph_value_info_in,
                graph_value_info_out
            )
            body_graph.value_info.extend(value_infos.values())
            # print("[loop subgraph]")
            # print(onnx.helper.printable_graph(body_graph))

        assert(len(trip_count.out_node) == 1)
        in_names = [trip_count.out_node[0], cond_in.name]
        for live_name in live_names:
            in_names.append(self.name_dict[live_name])
        out_names = []
        out_value_infos = {}

        for live_name in live_names:
            new_name = self.gen_name(live_name)
            new_value_info = value_info_with_name(
                self.value_info[self.name_dict[live_name]], new_name, self)
            out_value_infos[new_name] = new_value_info
            self.name_dict[live_name] = new_name
            out_names.append(new_name)

        loop_node = onnx.helper.make_node(
            'Loop',
            inputs=in_names,
            outputs=out_names,
            body=body_graph
        )

        ret_node.set_outputs([loop_node], out_names, out_value_infos)

        return ret_node
    
    def visit_While(self, node):  # ASTNode -> OnnxNodes
        if not isinstance(node.test, ast.Name): raise NotImplementedError
        cond = self.visit(node.test)
        assert(len(cond.out_node) == 1)
        live_names = self.ast_info[node]
        live_names = [node.test.id] + list(filter(lambda x: x != node.test.id, live_names))

        with self.new_graph_space():
            iter_count_name = self.gen_name('iter_count')
            iter_count = onnx.helper.make_tensor_value_info(iter_count_name, onnx.TensorProto.INT64, [])

            cond_in_name = self.gen_name('cond_in')
            cond_in = onnx.helper.make_tensor_value_info(cond_in_name, onnx.TensorProto.BOOL, [])

            graph_value_info_in = [iter_count, cond_in]
            for live_name in live_names:
                if live_name == node.test.id: continue
                name_in = self.gen_name(live_name)
                self.try_add_argument(live_name)
                value_info = value_info_with_name(
                    self.value_info[self.name_dict[live_name]], name_in, self)
                self.name_dict[live_name] = name_in
                graph_value_info_in.append(value_info)

            compute_nodes = []
            value_infos = {node.test.id: cond_in}
            for stmt in node.body:
                onnx_nodes = self.visit(stmt)
                compute_nodes += onnx_nodes.def_nodes
                value_infos.update(onnx_nodes.def_value_infos)

            graph_value_info_out = [value_infos[self.name_dict[node.test.id]]]
            for live_name in live_names:
                if live_name == node.test.id: continue
                graph_value_info_out.append(
                    self.value_info[self.name_dict[live_name]])

            body_graph = onnx.helper.make_graph(
                compute_nodes,
                self.gen_name('WHILE_LOOP'),
                graph_value_info_in,
                graph_value_info_out
            )
            body_graph.value_info.extend(value_infos.values())
            print("[while subgraph]")
            print(onnx.helper.printable_graph(body_graph))

        ret_node = OnnxNodes()
        in_names = [""]
        for live_name in live_names:
            in_names.append(self.name_dict[live_name])
        out_names = []
        out_value_infos = {}

        for live_name in live_names:
            if live_name == node.test.id: continue
            new_name = self.gen_name(live_name)
            new_value_info = value_info_with_name(
                self.value_info[self.name_dict[live_name]], new_name, self)
            out_value_infos[new_name] = new_value_info
            self.name_dict[live_name] = new_name
            out_names.append(new_name)
        self.name_dict.pop(node.test.id) # will cause key error if other parts of the program uses the condition. If that happends, we need to add a new node to set the condition to false.
        loop_node = onnx.helper.make_node(
            'Loop',
            inputs=in_names,
            outputs=out_names,
            body=body_graph
        )
        ret_node.set_outputs([loop_node], out_names, out_value_infos)
        return ret_node


    def visit_If(self, node):
        cond = self.visit(node.test)
        assert(len(cond.out_node) == 1)
        body_graphs = []
        name_dict_live = {}
        live_names = self.ast_info[node]
        for live_name in live_names:
            name_dict_live[live_name] = []
        name_dict_old = self.name_dict.copy()
        # start contextmanager
        for body_stmts in (node.body, node.orelse):
            self.name_dict = name_dict_old.copy()
            graph_value_info_in = []
            # for live_name in live_names:
            #     name_in = self.gen_name(live_name)
            #     self.try_add_argument(live_name)
            #     value_info = value_info_with_name(
            #         self.value_info[self.name_dict[live_name]], name_in, self)
            #     self.name_dict[live_name] = name_in
            #     graph_value_info_in.append(value_info)
            
            compute_nodes = []
            value_infos = {}
            for stmt in body_stmts:
                onnx_nodes = self.visit(stmt)
                compute_nodes += onnx_nodes.def_nodes
                value_infos.update(onnx_nodes.def_value_infos)

            node_names = set()
            for compute_node in compute_nodes:
                node_names.update(compute_node.output)

            graph_value_info_out = []
            for live_name in live_names:
                self.try_add_argument(live_name)
                if self.name_dict[live_name] not in node_names:
                    name = self.gen_name(live_name)
                    compute_nodes.append(
                        onnx.helper.make_node(
                            'Identity',
                            inputs = [self.name_dict[live_name]],
                            outputs = [name]
                        )
                    )
                    new_value_info = copy.deepcopy(self.value_info[self.name_dict[live_name]])
                    new_value_info.name = name
                    graph_value_info_out.append(new_value_info)
                else:
                    graph_value_info_out.append(
                        self.value_info[self.name_dict[live_name]])

            body_graph = onnx.helper.make_graph(
                compute_nodes,
                self.gen_name('IF_NODE'),
                graph_value_info_in,
                graph_value_info_out
            )
            body_graph.value_info.extend(value_infos.values())
            # print("[if subgraph]")
            # print(onnx.helper.printable_graph(body_graph))
            body_graphs.append(body_graph)
            for live_name in live_names:
                name_dict_live[live_name].append(self.name_dict[live_name])
            name_dict_cur = self.name_dict.copy()
            self.name_dict = name_dict_old.copy()
            for name, valueInfo in self.write_through.items():
                if name in name_dict_cur and name not in self.name_dict:
                    self.name_dict[name] = valueInfo.name
                    name_dict_old[name] = valueInfo.name

        # TODO: unify the types in name_dict_live
        out_names = []
        out_value_infos = {}
        for live_name in live_names:
            new_name = self.gen_name(live_name)
            new_value_info = value_info_with_name(
                self.value_info[name_dict_live[live_name][0]], new_name, self)
            out_value_infos[new_name] = new_value_info
            self.name_dict[live_name] = new_name
            out_names.append(new_name)

        if_node = onnx.helper.make_node(
            'If',
            inputs = [cond.out_node[0]],
            outputs = out_names,
            then_branch = body_graphs[0],
            else_branch = body_graphs[1],
        )

        cond.set_outputs([if_node], out_names, out_value_infos)
        return cond


    def visit_Pass(self, node):  # ASTNode -> OnnxNodes
        return OnnxNodes()

    def visit_Tuple(self, node):
        if isinstance(node.ctx, ast.Load):
            ret = OnnxNodes()
            out_names = []
            for sub_node in node.elts:
                assert isinstance(node.ctx, ast.Load)
                elem_node = self.visit(sub_node)
                ret = ret + elem_node
                assert len(elem_node.out_node) == 1
                out_names.append(elem_node.out_node[0])
            ret.set_outputs([], out_names)
            return ret
        else:
            raise NotImplementedError

    def visit_UnaryOp(self, node):
        op = self.visit(node.operand)
        op_to_onnx = {
            ast.USub: 'Neg',
            ast.Not: 'Not',
        }
        if type(node.op) in op_to_onnx:
            name = self.get_or_gen_name(node)
            onnx_node = onnx.helper.make_node(
                op_to_onnx[type(node.op)],
                inputs=op.out_node,
                outputs=[name],
            )
            value_info = type_to_value_info(name, self.get_type_of_node(node), self)
            op.set_outputs([onnx_node], [name], {name:value_info}) 
            return op
        else:
            print(astunparse.dump(node))
            raise NotImplementedError


    def visit_Compare(self, node):
        # print(astunparse.dump(node))
        # exit(1)
        left_node = self.visit(node.left)
        assert(len(node.comparators) == 1)
        right_node = self.visit(node.comparators[0])
        op2onnx = {
            ast.Eq: 'Equal',
            ast.Gt: 'Greater',
            ast.Lt: 'Less',
        }
        ret = left_node + right_node
        name = self.get_or_gen_name(node)
        onnx_node = onnx.helper.make_node(
            op2onnx[type(node.ops[0])],
            inputs=[left_node.out_node[0], right_node.out_node[0]],
            outputs = [name],
        )
        value_info = type_to_value_info(name, self.get_type_of_node(node), self)
        ret.set_outputs([onnx_node], [name], {name:value_info}) 
        return ret


    def visit(self, node):  # ASTNode -> OnnxNodes / GraphProto / ModelProto
        # print("ast.visit:", node, astunparse.unparse(node))
        if isinstance(node, ast.FunctionDef):
            return self.visit_FunctionDef(node)
        elif isinstance(node, ast.Assign):
            return self.visit_Assign(node)
        elif isinstance(node, ast.Module):
            return self.visit_Module(node)
        elif isinstance(node, ast.Return):
            return self.visit_Return(node)
        elif isinstance(node, ast.Call):
            return self.visit_Call(node)
        elif isinstance(node, ast.Attribute):
            return self.visit_Attribute(node)
        elif isinstance(node, ast.BinOp):
            return self.visit_BinOp(node)
        elif isinstance(node, ast.Name):
            return self.visit_Name(node)
        elif isinstance(node, ast.Subscript):
            return self.visit_Subscript(node)
        elif isinstance(node, ast.Constant):
            return self.visit_Constant(node)
        elif isinstance(node, ast.For):
            return self.visit_For(node)
        elif isinstance(node, ast.If):
            return self.visit_If(node)
        elif isinstance(node, ast.Pass):
            return self.visit_Pass(node)
        elif isinstance(node, ast.Tuple):
            return self.visit_Tuple(node)
        elif isinstance(node, ast.UnaryOp):
            return self.visit_UnaryOp(node)
        elif isinstance(node, ast.Compare):
            return self.visit_Compare(node)
        elif isinstance(node, ast.While):
            return self.visit_While(node)
        else:
            raise NotImplementedError(
                "AST type not supported: {} {}".format(node, astunparse.unparse(node)))
