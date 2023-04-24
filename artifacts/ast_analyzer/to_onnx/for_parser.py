import gast as ast
from onnx import helper
from .node import OnnxNodes
from .utils import type_to_value_info
import astunparse

def get_for_parser(target_node, iter_node):
    if isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name) and iter_node.func.id == 'range':
        return ForRange(target_node, iter_node)
    else:
        raise NotImplementedError


class ForRange():
    def __init__(self, target_node, iter_node):
        self.target_node = target_node
        self.iter_node = iter_node

    def trip_count(self, visitor):  # type: ExportEngine -> OnnxNodes
        if len(self.iter_node.args) == 1:
            node_ty = visitor.get_type_of_node(self.iter_node.args[0])
            assert(node_ty.is_int())
            return visitor.visit(self.iter_node.args[0])
        elif len(self.iter_node.args) == 3: # not tested
            start_ty = visitor.get_type_of_node(self.iter_node.args[0])
            end_ty = visitor.get_type_of_node(self.iter_node.args[1])
            assert(start_ty.is_int())
            assert(end_ty.is_int())
            start_node = visitor.visit(self.iter_node.args[0])
            assert(len(start_node.out_node) == 1)
            end_node = visitor.visit(self.iter_node.args[1])
            assert(len(end_node.out_node) == 1)
            if isinstance(self.iter_node.args[2], ast.Constant) and eval(astunparse.unparse(self.iter_node.args[2])) in (-1, 1):
                step_val = eval(astunparse.unparse(self.iter_node.args[2]))
                cnt_name = visitor.gen_name()
                if step_val == 1:
                    cnt_onnx_node = helper.make_node('Sub', [end_node.out_node[0], start_node.out_node[0]], cnt_name)
                else:
                    assert(step_val == -1)
                    cnt_onnx_node = helper.make_node('Sub', [start_node.out_node[0], end_node.out_node[0]], cnt_name)
                cnt_value_info = type_to_value_info(cnt_name, start_ty, visitor)
                count_node = OnnxNodes()
                count_node += start_node
                count_node += end_node
                count_node.set_output(cnt_onnx_node, cnt_name, cnt_value_info)
                return count_node
        raise NotImplementedError

    # type: ValueInfoProto, ExportEngine -> OnnxNodes
    def get_target(self, iter_count, visitor):
        if len(self.iter_node.args) == 1:
            val_nodes = OnnxNodes([], [iter_count.name], {})
        else:
            raise NotImplementedError

        if isinstance(self.target_node, ast.Name):
            visitor.name_dict[self.target_node.id] = val_nodes.out_node[0]
        else:
            raise NotImplementedError

        return val_nodes
