import gast as ast
import onnx
import astunparse

from .anf import anf
from . import quoting
from .reverse_ad import reverse_ad
from .annotate import resolve_calls, mark_shape
from .simplify import simplify


def grad_func(func, type_dict):
    node = func.__ast__
    resolve_calls(node, func)
    anf_node = anf(node)
    # print(quoting.to_source(anf_node))
    new_fwd, new_bwd = reverse_ad(anf_node.body[0]).body
    # print(astunparse.unparse(new_fwd))
    # print(astunparse.unparse(new_bwd))
    raise NotImplementedError


def grad_model(model, type_dict, device):
    node = model.__ast__
    resolve_calls(node, model.forward)
    mark_shape(node, type_dict)
    anf_node = anf(node)
    # print("[ANF]")
    # print(quoting.to_source(anf_node))
    new_fwdbwd, attrs_order = reverse_ad(anf_node.body[0], device)
    # print("[Codegen]")
    # print(astunparse.unparse(new_fwdbwd))
    simplify(new_fwdbwd, device)
    # print("[Simplify]")
    # print(astunparse.unparse(new_fwdbwd))
    return new_fwdbwd, attrs_order