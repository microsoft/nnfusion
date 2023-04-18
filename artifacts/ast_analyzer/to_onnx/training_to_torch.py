
from ast_analyzer.utils.unparse import unparse_ast_list
import copy
import gast
import astunparse
from ast_analyzer.grad import annotations as anno
from ast_analyzer.shape_inference.type_inference import *
import os

def to_torch_code(model_name, n_rets, node, attrs_input, attrs_save, node2type_fwd, node2type_bwd):
    assert(isinstance(node.body[0].body[-1], gast.Return))
    assert(isinstance(node.body[1].body[-1], gast.Return))

    attr_nodes = [None for _ in attrs_input]
    attr_name_in_func = [None for _ in attrs_input]
    fwd_body_nodes = []
    for stmt in node.body[0].body[:-1]:
        if isinstance(stmt, gast.Assign) and isinstance(stmt.value, gast.Attribute):
            idx = attrs_input.index(stmt.value.attr)
            attr_nodes[idx] = copy.deepcopy(stmt.value)
            attr_name_in_func[idx] = stmt.targets[0].id
        else:
            fwd_body_nodes.append(stmt)

    for x in attr_nodes: assert(x is not None)
    for x in attr_name_in_func: assert(x is not None)

    fwd_arg_names = [x.id for x in node.body[0].args.args if x.id != 'self']
    fwd_arg_names.extend(attr_name_in_func)

    bwd_arg_names = [x.id for x in node.body[1].args.args if x.id != 'self']

    dir_name = os.path.dirname(os.path.abspath(__file__))
    template = []

    with open(dir_name + '/class_template_train.py') as f:
        for st in f.readlines():
            template.append(st)
    gen = []

    fwd_ret_ids = []
    for name_node in node.body[0].body[-1].value.elts:
        fwd_ret_ids.append(name_node.id)

    for st in template:
        if "^^FWD_INPUTS" in st:
            st = st.replace("^^FWD_INPUTS", ", ".join(fwd_arg_names))
            gen.append(st)
        elif "^^FWD_CODE" in st:
            codes = unparse_ast_list(fwd_body_nodes)
            codes = codes.split("\n")
            for c in codes:
                if len(c) == 0: continue
                s = st.replace("^^FWD_CODE", c)
                gen.append(s)
        elif "^^CTX_SAVE" in st:
            st = st.replace("^^CTX_SAVE", ", ".join(fwd_ret_ids[n_rets:] + attrs_save))
            gen.append(st)
        elif "^^FWD_RETURN" in st:
            st = st.replace("^^FWD_RETURN", ", ".join(fwd_ret_ids[:n_rets]))
            gen.append(st)
        elif "^^BWD_INPUTS" in st:
            st = st.replace("^^BWD_INPUTS", ", ".join(bwd_arg_names[:n_rets]))
            gen.append(st)
        elif "^^CTX_OR_SAVE" in st:
            st = st.replace("^^CTX_OR_SAVE", ", ".join(bwd_arg_names[n_rets:]))
            gen.append(st)
        elif "^^BWD_CODE" in st:
            codes = unparse_ast_list(node.body[1].body[:-1])
            codes = codes.split("\n")
            for c in codes:
                if len(c) == 0: continue
                s = st.replace("^^BWD_CODE", c)
                gen.append(s)
        elif "^^BWD_RETURN_STMT" in st:
            st = st.replace("^^BWD_RETURN_STMT", astunparse.unparse(node.body[1].body[-1]).replace("\n", ""))
            gen.append(st)
        elif "^^SCALAR_TO_TENSOR" in st:
            ret_type = node2type_fwd[node.body[0].body[-1]]
            for nd, ty in zip(node.body[0].body[-1].value.elts, ret_type):
                if not isinstance(ty, TyTensor):
                    assert(isinstance(nd, gast.Name))
                    s = st.replace("^^SCALAR_TO_TENSOR", f"{nd.id} = torch.tensor({nd.id})")
                    gen.append(s)
        elif "^^TENSOR_TO_SCALAR" in st:
            for arg_node in node.body[1].args.args:
                if isinstance(node2type_bwd[arg_node], TyNum):
                    assert(isinstance(arg_node, gast.Name))
                    s = st.replace("^^TENSOR_TO_SCALAR", f"{arg_node.id} = {arg_node.id}.item()")
                    gen.append(s)
        else:
            gen.append(st)

    with open('tmp/Model' + model_name + '_train.py', 'w') as f:
        for st in gen:
            f.write(st)

    arg_nodes = []
    for arg in node.body[0].args.args:
        if arg.id != 'self':
            arg_nodes.append(gast.Name(id=arg.id, ctx=gast.Load(), annotation=None, type_comment=None))
    return arg_nodes + attr_nodes