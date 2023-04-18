# grad style
from ast_analyzer.grad import ast_utils
from ast_analyzer.grad import annotations as anno
from ast_analyzer.grad import template
from ast_analyzer.shape_inference.types import *
from ast_analyzer.shape_inference.shape_elem import unwrap_shape
import gast

class FillUnbroadcast(gast.NodeTransformer):
    def extract_type(self, node):
        if anno.hasanno(node, 'type'):
            return anno.getanno(node, 'type')
        return None

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, gast.Attribute) and node.func.attr == "unbroadcast":
            src = self.extract_type(node.args[0])
            dst = self.extract_type(node)
            if dst is None:
                dst = self.extract_type(node.args[1])
            if src is None:
                raise ValueError("can not extract src type from {} {}".format(
                    astunparse.unparse(node), astunparse.dump(node)))
            if dst is None:
                raise ValueError("can not extract dst type from {} {}".format(
                    astunparse.unparse(node), astunparse.dump(node)))
            if isinstance(src, TyTensor) and isinstance(dst, TyTensor):
                if is_sametype(src, dst):
                    return node.args[0]
                src_shape = unwrap_shape(src.shape)
                dst_shape = unwrap_shape(dst.shape)
                if len(src_shape) < len(dst_shape):
                    raise ValueError(
                        "cannnot unbroadcast: src.shape = {}, dst.shape = {}".format())
                to_reduce = []
                keepdim = False
                for i, j in zip(range(len(src_shape)), range(len(dst_shape) - len(src_shape), len(dst_shape))):
                    if j < 0:
                        to_reduce.append(i)
                        continue
                    if dst_shape[j] is None:
                        return node
                    if dst_shape[j] == 1 and src_shape[i] != 1:
                        to_reduce.append(i)
                        keepdim = True

                if len(to_reduce) > 0:
                    reduce_ = template.replace("torch.sum(x, dim = {}, keepdim = {})".format(
                        tuple(to_reduce), keepdim), x=ast_utils.copy_node(node.args[0])).value
                else:
                    reduce_ = ast_utils.copy_node(node.args[0])

                if len(src_shape) != len(dst_shape) and keepdim:
                    for i in range(len(src_shape) - len(dst_shape)):
                        reduce_ = template.replace(
                            "torch.sequeeze(x, dim=0)", x=reduce_)

                return reduce_
        return node


def fill_shape(node):
    FillUnbroadcast().visit(node)
