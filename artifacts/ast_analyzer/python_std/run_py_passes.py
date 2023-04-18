import inspect
import gast
import ast
from ast_analyzer.shape_inference.utils import clip_head
import astunparse
from ast_analyzer.python_std.passmanager import PassManager
from ast_analyzer.python_std.optimizations import *
from ast_analyzer.python_std.analyses import *
import torch
from ast_analyzer.grad.annotate import block_live
from ast_analyzer.grad.annotate import mark_shape

RUN_DYNAMIC_UNROLL = True


def apply_passes_ast(node, obj):
    assert(isinstance(node, gast.Module))
    pm = PassManager("opt", obj=obj)
    # annotations = {}
    # for arg in node.body[0].args.args:
    #     if arg.annotation is not None:
    #         annotations[arg] = arg.annotation
    #         arg.annotation = None
    # _, node = pm.apply(ExpandBuiltins, node)
    _, node = pm.apply(ReplaceObjConst, node)
    if RUN_DYNAMIC_UNROLL:
        _, node = pm.apply(DynamicLoopSplit, node)
        _, node = pm.apply(ListToNode, node)
    _, node = pm.apply(ToTorchTransform, node)
    _, node = pm.apply(LoopFullUnrolling, node)
    _, node = pm.apply(ListToNode, node)
    _, node = pm.apply(ToTensorTransform, node)
    _, node = pm.apply(CopyToAssign, node)
    # for i in range(2):
    #     _, node = pm.apply(PatternTransform, node)
    #     _, node = pm.apply(Functional, node)
    #     _, node = pm.apply(ForwardSubstitution, node)
    #     _, node = pm.apply(ConstantFolding, node)
    #     _, node = pm.apply(PartialConstantFolding, node)
    #     _, node = pm.apply(ConstantPropagation, node)
    #     _, node = pm.apply(DeadCodeElimination, node)
    #     _, node = pm.apply(ToTensorTransform, node)
    # for arg, annotation in annotations.items():
    #     arg.annotation = annotation
    print("[Python Compile]")
    print(astunparse.unparse(node))
    return node


def disable_dynamic_unroll():
    global RUN_DYNAMIC_UNROLL
    RUN_DYNAMIC_UNROLL = False


def apply_passes(func, obj):
    code = clip_head(inspect.getsource(func))
    node = gast.ast_to_gast(ast.parse(code))
    return apply_passes_ast(node, obj)


def run(func, model):
    print("run by passes", func)
    node = apply_passes(func, None)
    func.__ast__ = node

    # apply_passes(model)
    for submodel in model.modules():
        print("run by passes", type(submodel))
        if type(submodel) not in torch.nn.__dict__.values():
            node = apply_passes(submodel.forward, submodel)
            submodel.__ast__ = node


def run_function(func):
    print("run function", func)
    node = apply_passes(func, None)
    func.__ast__ = node


def run_model(model):
    for submodel in model.modules():
        if type(submodel) not in torch.nn.__dict__.values():
            print("run model", type(submodel))
            node = apply_passes(submodel.forward, submodel)
            submodel.__ast__ = node
            if hasattr(submodel, "forward_full"):
                # TODO: a cleverer way to support it
                print("run forward full", type(submodel))
                node = apply_passes(submodel.forward_full, submodel)
                submodel.forward_full.__ast__ = node
    return model.__ast__


def apply_passes_pre_onnx(node, obj):
    pm = PassManager("pre_onnx", obj=obj)
    _, node = pm.apply(NormalizeIfElse, node)
    _, node = pm.apply(FillIfDef, node)
    return node


def gather_info_pre_onnx(node):
    return block_live(node)


def pre_onnx_model(model, ast_attr):
    for submodel in model.modules():
        if type(submodel) not in torch.nn.__dict__.values():
            # print("pre_onnx", type(submodel))
            node = apply_passes_pre_onnx(getattr(submodel, ast_attr), submodel)
            setattr(submodel, ast_attr, node)
            ast_info = gather_info_pre_onnx(node)
            submodel.__ast_info__ = ast_info


def fill_shape_func(node, node2type):
    from ast_analyzer.grad.annotate import mark_shape
    mark_shape(node, node2type)
    fill_shape(node)
    # print("[fillshape]", astunparse.unparse(node))


def apply_passes_recursive(node, node2type, called_functions):
    mark_shape(node, node2type)

    pm = PassManager("opt", obj=None) # TODO: obj
    # print("[start]")
    _, node = pm.apply(ReplaceInferedConst, node)
    _, node = pm.apply(DeadCodeElimination, node)
    _, node = pm.apply(UnrollTuple, node)
    _, node = pm.apply(ListToNode, node)
    # put unrollsequential as late as possible, as the generated nodes do not have 'type' attribute.
    _, node = pm.apply(UnrollSequential, node)
    _, node = pm.apply(ListToNode, node)

    # print("[python opt]")
    # print(astunparse.unparse(node))

    for subnode in gast.walk(node):
        if isinstance(subnode, gast.Call) and getattr(subnode, 'is_udf', False):
            func_inst = subnode._func_inst
            assert(func_inst is not None)
            # print("[recursive]", astunparse.unparse(subnode), func_inst)
            if func_inst not in called_functions:
                called_functions.add(func_inst)
                if isinstance(func_inst, torch.nn.Sequential):
                    for func_node in subnode.func_nodes:
                        if func_node is not None:
                            apply_passes_recursive(func_node, node2type, called_functions)
                else:
                    apply_passes_recursive(subnode.func_node, node2type, called_functions)
                called_functions.remove(func_inst)