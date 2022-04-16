from glob import glob
import tvm
import re
import inspect
from tvm import te
from test_config import *

'''
rewrite_expr fuse contigous axis with the same type of the orginal expr,
and return the new expr
to use rewrite_expr, the original expr must follow the rules below:
    1. all axis vars should be named in lower case
    2. the axis/tensor name should be the same with the axis/tensor var
    3. the axis length var is upper case one of the axis var
    4. the shape argument should be unpacked in the expr

    for example:
        right expr:
            def conv_expr(shape, for_rtile=False, pad={}):
            N, F, HO, WO, C, KH, KW = shape
            S, P, D = 1, 0, 1
            H = (HO - 1) * S + KH - 2 * P
            W = (WO - 1) * S + KW - 2 * P

            data = te.placeholder((N, C, H, W), name="data")
            kernel = te.placeholder((F, C, KH, KW), name="kernel")

            c = te.reduce_axis((0, C), name='c')
            kh = te.reduce_axis((0, KH), name='kh')
            kw = te.reduce_axis((0, KW), name='kw')

            conv = te.compute((N, F, HO, WO), lambda n, f, ho, wo:\
                        te.sum(data[n, c, ho * S + kh, wo * S + kw] *
                                kernel[f, c, kh, kw],
                                axis=[c, kh, kw])
                                , name='conv')

            return [data, kernel], [conv]
        
        wrong expr1:
            def conv_expr(shape, for_rtile=False, pad={}):
            N, F, HO, WO, CI, KH, KW = shape    # axis is c, while the length is CI
            S, P, D = 1, 0, 1
            H = (HO - 1) * S + KH - 2 * P
            W = (WO - 1) * S + KW - 2 * P

            data = te.placeholder((N, CI, H, W), name="data1") # tensor var is data, while the name is data1
            kernel = te.placeholder((F, CI, KH, KW), name="kernel")

            c = te.reduce_axis((0, CI), name='c')
            kh = te.reduce_axis((0, KH), name='kh')
            kw = te.reduce_axis((0, KW), name='kw')

            conv = te.compute((N, F, HO, WO), lambda n, f, ho, wo:\
                        te.sum(data[n, c, ho * S + kh, wo * S + kw] *
                                kernel[f, c, kh, kw],
                                axis=[c, kh, kw])
                                , name='conv')

            return [data, kernel], [conv]

        wrong expr2:
            def add_expr(shape, for_rtile=False, pad={}):
                A = te.placeholder(shape, name="A") # the shape argument is not unpacked
                B = te.placeholder(shape, name="B")
                C = te.compute(shape, lambda *i: A(*i) + B(*i), name="compute") # tensor var is C, while the name is compute
                return [A, B], [C]
'''


def rewrite_expr(expr, shape, new_expr_name, transpose=True):
    expr_out = expr(shape)
    ins, outs = expr_out[0], expr_out[1]
    out = outs[0]
    out_name = out.name # assume only one output
    ins_name = [i.name for i in ins]
    saxis = out.op.axis # out axis
    raxis = out.op.reduce_axis
    saxis_name = [a.var.name for a in saxis]
    raxis_name = [a.var.name for a in raxis]
    loop_order = saxis_name + raxis_name
    axis_set = set(loop_order)           

    expr_str = inspect.getsource(expr) # original expr

    # find output compute expr
    match = '(    | )' + out_name + ' ?= ?te\.compute\((.|\n)+name ?= ?(\'|\")' + out_name + '(\'|\")\)'
    x = re.search(match, expr_str)
    assert x
    out_expr = x.group()
    assert out_expr != ''
    indent = out_expr[: out_expr.find(out_name)]
    # find original inputs indices and axis in output compute expr
    ins_str = ["" for i in range(len(ins_name))]
    for i in range(len(ins_name)):
        start = out_expr.find(ins_name[i] + '[')
        count = 1
        end = start
        for j in range(start + len(ins_name[i]) + 1, len(out_expr)):
            if count == 0:
                end = j
                break
            if out_expr[j] == '[':
                count += 1
            elif out_expr[j] == ']':
                count -= 1
        in_str = out_expr[start: end]
        ins_str[i] = in_str

    ins_axis = [[] for _ in range(len(ins_str))]
    ins_axis_split = [[] for _ in range(len(ins_str))]
    for i in range(len(ins_str)):
        in_str = ins_str[i].replace(' ', '')
        axes = in_str[len(ins_name[i]) + 1: -1].split(',')
        for a in axes:
            a = re.split(r'([-+*/()%])|\s+', a)
            ins_axis_split[i].append(a)
            for a_i in a:
                if a_i in axis_set:
                    ins_axis[i].append(a_i)
     
    updated_ins_axis = [[] for _ in range(len(ins_str))]
    in_axis_dict = [{} for _ in range(len(ins_str))]
    out_axis_dict = {saxis_name[i]: i for i in range(len(saxis_name))}
    all_axis_dict = in_axis_dict + [out_axis_dict]

    # reorder inputs axis based on loop order
    for i in range(len(ins_axis)):
        in_axis = ins_axis[i]
        in_axis_set = set(in_axis)
        idx = 0
        for a in loop_order:
            if a in in_axis_set:
                updated_ins_axis[i].append(a)
                in_axis_dict[i][a] = idx
                idx += 1

    # fuse axis with rule: contiguous axes with the same type in all tensors that contains these axes
    sa_fusable = []
    sa_fused_axis = {}
    for i in range(len(saxis_name) - 1):
        if len(sa_fusable) > 0 and sa_fusable[-1][-1] - 1 >= i:
            continue
        cur = saxis_name[i]
        cand = []
        exclude = []
        for j in range(len(all_axis_dict)):
            if cur in all_axis_dict[j]:
                cand.append(all_axis_dict[j])
            else:
                exclude.append(all_axis_dict[j])
        n = i + 1
        fusable = True
        while n < len(saxis_name) and fusable:
            nxt = saxis_name[n]
            for e in range(len(exclude)):
                if nxt in exclude[e]:
                    fusable = False
                    break
            for c in range(len(cand)):
                if nxt not in cand[c]: 
                    fusable = False
                    break
                if cand[c][nxt] != cand[c][cur] + 1:
                    fusable = False
                    break          
            cur = nxt
            n += 1
        if fusable:
            sa_fusable.append((i, n))
            sa_fused_axis['sa_fused' + str(len(sa_fused_axis))] = saxis_name[i: n]

    ra_fusable = []
    ra_fused_axis = {}
    for i in range(len(raxis_name) -1):
        if len(ra_fusable) > 0 and ra_fusable[-1][-1] - 1 >= i:
            continue
        cur = raxis_name[i]
        cand = []
        exclude = []
        for j in range(len(all_axis_dict)):
            if cur in all_axis_dict[j]:
                cand.append(all_axis_dict[j])
            else:
                exclude.append(all_axis_dict[j])
        n = i + 1
        fusable = True
        while n < len(raxis_name):
            nxt = raxis_name[n]
            for e in range(len(exclude)):
                if nxt in exclude[e]:
                    fusable = False
                    break
            for c in range(len(cand)):
                if nxt not in cand[c]: 
                    fusable = False
                    break
                if cand[c][nxt] != cand[c][cur] + 1:
                    fusable = False
                    break
            cur = nxt
            n += 1
        if fusable:
            ra_fusable.append((i, n))
            ra_fused_axis['ra_fused' + str(len(ra_fused_axis))] = raxis_name[i: n]
    
    if len(sa_fused_axis) == 0 and len(ra_fused_axis) == 0:
        print("expr axis is not fusable")
        return expr
    all_axis = updated_ins_axis + [saxis_name]
    fused_axis = [[] for _ in range(len(all_axis))]

    for j in range(len(all_axis)):
        axis = all_axis[j]
        i = 0
        while i < len(axis):
            has_fused = False
            for s in sa_fused_axis:
                if axis[i: i + len(sa_fused_axis[s])] == sa_fused_axis[s]:
                    fused_axis[j].append(s)
                    i += len(sa_fused_axis[s])
                    has_fused = True
                    break
            for r in ra_fused_axis:
                if axis[i: i + len(ra_fused_axis[r])] == ra_fused_axis[r]:
                    fused_axis[j].append(r)
                    i += len(ra_fused_axis[r])
                    has_fused = True
                    break
            if not has_fused:
                fused_axis[j].append(axis[i])
                i += 1
    if transpose:
        reverse_sa_fused_map = {}
        reverse_ra_fused_map = {}
        for a in sa_fused_axis:
            for ori in sa_fused_axis[a]:
                reverse_sa_fused_map[ori] = a
        for a in ra_fused_axis:
            for ori in ra_fused_axis[a]:
                reverse_ra_fused_map[ori] = a

        updated_saxis_name = []
        updated_raxis_name = []
        for a in saxis_name:
            if a in reverse_sa_fused_map:
                fused = reverse_sa_fused_map[a]
            else:
                fused = a
            if len(updated_saxis_name) == 0 or updated_saxis_name[-1] != fused:
                updated_saxis_name.append(fused)
        for a in raxis_name:
            if a in reverse_ra_fused_map:
                fused = reverse_ra_fused_map[a]
            else:
                fused = a
            if len(updated_raxis_name) == 0 or updated_raxis_name[-1] != fused:
                updated_raxis_name.append(fused)

        fused_axis_set = [set(fa) for fa in fused_axis]
        updated_sa_fused_axis = sa_fused_axis.copy()
        updated_ra_fused_axis = ra_fused_axis.copy()

        i = 0
        j = 2
        while j - i > 1 and j < len(updated_saxis_name):
            ai = updated_saxis_name[i]
            aj = updated_saxis_name[j]

            cache_idx = []
            fusable = False
            for s in range(len(fused_axis_set)):
                fs = fused_axis_set[s]
                if fusable:
                    if (ai in fs and aj not in fs) or (ai not in fs and aj in fs):
                        fusable = False
                        break
                    elif ai in fs and aj in fs:
                        cache_idx.append(s)
                else:
                    if ai in fs and aj in fs:
                        fusable = True
                        cache_idx.append(s)
            
            if fusable:
                ori = []
                if ai in updated_sa_fused_axis:
                    ori.extend(updated_sa_fused_axis[ai])
                else:
                    ori.append(ai)
                if aj in updated_sa_fused_axis:
                    ori.extend(updated_sa_fused_axis[aj])
                else:
                    ori.append(aj)
                fused_name = ai + '_' + aj      
                updated_sa_fused_axis[fused_name]  = ori
                updated_sa_fused_axis.pop(ai, None)
                updated_sa_fused_axis.pop(aj, None)
                updated_saxis_name[j] = fused_name
                updated_saxis_name.pop(i)

                for idx in cache_idx:
                    fs = fused_axis[idx]
                    fs.remove(ai)
                    for t in range(len(fs)):
                        if fs[t] == aj:
                            fs[t] = fused_name
                            break                    

                    fs_set = fused_axis_set[idx]
                    fs_set.discard(ai)
                    fs_set.discard(aj)
                    fs_set.add(fused_name)
            else:
                if j < len(updated_saxis_name):
                    j += 1
                else:
                    i += 1
                    j += 1

        i = 0
        j = 2
        while j - i > 1 and j < len(updated_raxis_name):
            ai = updated_raxis_name[i]
            aj = updated_raxis_name[j]

            cache_idx = []
            fusable = False
            for s in range(len(fused_axis_set)):
                fs = fused_axis_set[s]
                if fusable:
                    if (ai in fs and aj not in fs) or (ai not in fs and aj in fs):
                        fusable = False
                        break
                    elif ai in fs and aj in fs:
                        cache_idx.append(s)
                else:
                    if ai in fs and aj in fs:
                        fusable = True
                        cache_idx.append(s)
            
            if fusable:
                ori = []
                if ai in updated_ra_fused_axis:
                    ori.extend(updated_ra_fused_axis[ai])
                else:
                    ori.append(ai)
                if aj in updated_ra_fused_axis:
                    ori.extend(updated_ra_fused_axis[aj])
                else:
                    ori.append(aj)
                fused_name = ai + '_' + aj      
                updated_ra_fused_axis[fused_name]  = ori
                updated_ra_fused_axis.pop(ai, None)
                updated_ra_fused_axis.pop(aj, None)
                updated_raxis_name[j] = fused_name
                updated_raxis_name.pop(i)

                for idx in cache_idx:
                    fs = fused_axis[idx]
                    fs.remove(ai)
                    for t in range(len(fs)):
                        if fs[t] == aj:
                            fs[t] = fused_name
                            break

                    fs_set = fused_axis_set[idx]
                    fs_set.discard(ai)
                    fs_set.discard(aj)
                    fs_set.add(fused_name)
            else:
                if j < len(updated_raxis_name):
                    j += 1
                else:
                    i += 1
                    j += 1
            
        sa_fused_axis = updated_sa_fused_axis
        ra_fused_axis = updated_ra_fused_axis
        
    # get fused axis length and pad length
    fused_axis_len = {}
    fused_axis_len_list = {}

    for t in fused_axis:
        for a in t:
            if a not in fused_axis_len:
                if a in sa_fused_axis:
                    ori_axis = sa_fused_axis[a]
                    fused_axis_len[a] = ' * '.join([i.upper() for i in ori_axis])
                    fused_axis_len_list[a] = [i.upper() for i in ori_axis]
                elif a in ra_fused_axis:
                    ori_axis = ra_fused_axis[a]
                    fused_axis_len[a] = ' * '.join([i.upper() for i in ori_axis])
                    fused_axis_len_list[a] = [i.upper() for i in ori_axis]
                else:
                    fused_axis_len[a] = a.upper()
                    fused_axis_len_list[a] = [a.upper()]

    # replace old axis with fused axis
    fused_indice_map = {}
    for fa in sa_fused_axis:
        ori_axis = sa_fused_axis[fa]
        for i in range(len(ori_axis)):
            oa = ori_axis[i]
            if i == 0:
                fused_indice_map[oa] = fa + ' // (' + ' * '.join([a.upper() for a in ori_axis[i+1: ]]) + ')'
            elif i == len(ori_axis) - 1:
                pre = ori_axis[i - 1]
                fused_indice_map[oa] = '%'.join(fused_indice_map[pre].rsplit('//', 1))
            else:
                fused_indice_map[oa] = fa + ' % (' + ' * '.join([a.upper() for a in ori_axis[i: ]]) + ') // ' + '(' + ' * '.join([a.upper() for a in ori_axis[i+1: ]]) +')' 
    
    for fa in ra_fused_axis:
        ori_axis = ra_fused_axis[fa]
        for i in range(len(ori_axis)):
            oa = ori_axis[i]
            if i == 0:
                fused_indice_map[oa] = fa + ' // (' + ' * '.join([a.upper() for a in ori_axis[i+1: ]]) + ')'
            elif i == len(ori_axis) - 1:
                pre = ori_axis[i - 1]
                fused_indice_map[oa] = '%'.join(fused_indice_map[pre].rsplit('//', 1))
            else:
                fused_indice_map[oa] = fa + ' % (' + ' * '.join([a.upper() for a in ori_axis[i: ]]) + ') // ' + '(' + ' * '.join([a.upper() for a in ori_axis[i+1: ]]) +')' 
    
    # get padded input expr
    ins_pad_decl = '\n'
    ins_pad = ''

    fused_axis_len_pad = {}
    axis_pad = ''
    for a in fused_axis_len:
        l = fused_axis_len[a]
        pad_axis_name = a + '_pad'
        axis_pad += indent + pad_axis_name + ' = ' + l + '\n'
        axis_pad += indent + 'if \'' + a + "\' in pad: " + pad_axis_name + ' += pad[\'' + a + '\']\n'
        fused_axis_len_pad[a] = pad_axis_name
    ins_pad_decl += axis_pad  
    
    padded_ins_name = [n + '_pad' for n in ins_name]
    padded_ins_name_shape = []
    for i in range(len(ins_name)):
        name = ins_name[i]
        padded_name = padded_ins_name[i]
        f_axis = fused_axis[i]
        in_axis_split = ins_axis_split[i]
        shape_pad = [fused_axis_len_pad[a] for a in fused_axis[i]]
        shape = [fused_axis_len[a] for a in fused_axis[i]]
        padded_ins_name_shape.append((padded_name, shape_pad))
        condition = []
        for d in range(len(shape)):
            condition.append(f_axis[d] + ' < ' + shape[d])
        indice = []
        for s in in_axis_split:
            fused = []
            for a in s:
                if a in fused_indice_map:
                    fused.append(fused_indice_map[a])
                else:
                    fused.append(a)
            indice.append(' '.join(fused))
        
        in_pad = indent + '{} = te.compute([{}],\n'\
            '           lambda {}: te.if_then_else(te.all({}),\n'\
            '           {}[{}], 0.0),\n'\
            '           tag="{}", name="{}")\n'.format(
            padded_name, 
            ', '.join(shape_pad), 
            ', '.join(f_axis), 
            ', '.join(condition),
            name,
            ',\n                '.join(indice),
            padded_name,
            padded_name
            )

        ins_pad += in_pad + '\n'

    # get fused reduce axis
    raxis_fuse = ''
    for r in ra_fused_axis:
        ori_a = ra_fused_axis[r]
        r_fuse = indent + '{} = te.reduce_axis((0, {}), name="{}")\n'.format(
            r,
            ' * '.join([a.upper() for a in ori_a]),
            r
            )
        raxis_fuse += r_fuse + "\n"

    # get final output
    final_out_name = out_name + '_unpad'
    final_out_shape = [fused_axis_len[a] for a in fused_axis[-1]]
    final_out_expr = out_name +'[' + ', '.join(fused_axis[-1]) + ']'
    final_out = indent + '{} = te.compute([{}], lambda {}: {}, tag="{}", name="{}")\n'.format(
        final_out_name, 
        ', '.join(final_out_shape), 
        ', '.join(fused_axis[-1]), 
        final_out_expr,
        final_out_name,
        final_out_name
        )

    # replace old output compute expr with fused one
    out_shape_start = 'te.compute('
    out_shape_end = 'lambda '
    saxis_end = ':'

    out_shape_start_pos = out_expr.find(out_shape_start)
    assert out_shape_start_pos != -1
    out_shape_start_pos += len(out_shape_start)

    out_shape_end_pos = out_expr.find(out_shape_end)
    assert out_shape_end_pos != -1
    saxis_start_pos = out_shape_end_pos + len(out_shape_end)

    saxis_end_pos = out_expr.find(saxis_end)
    assert saxis_end_pos != -1

    out_shape = out_expr[out_shape_start_pos: out_shape_end_pos]
    out_axis = out_expr[saxis_start_pos: saxis_end_pos]

    out_fused_axis = fused_axis[-1]
    out_fused_shape = [fused_axis_len_pad[a] for a in out_fused_axis]
    new_out_expr = out_expr.replace(out_shape, '[' + ', '.join(out_fused_shape) + '], ', 1)\
                            .replace(out_axis, ', '.join(out_fused_axis), 1)

    for i in range(len(ins_str)):
        new_out_expr = new_out_expr.replace(ins_str[i], padded_ins_name[i] + '[' +', '.join(fused_axis[i]) + ']')

    raxis_match = 'axis ?= ?\[.+\]'
    m = re.search(raxis_match, new_out_expr)
    if m:
        raxis_str = m.group()
        new_out_expr = new_out_expr.replace(raxis_str, 'axis=[' + ', '.join(list(ra_fused_axis.keys())) + ']')

    # find unpack shape line
    unpack_shape_match = re.search(indent + '.+ ?= ?shape\n', expr_str)
    assert unpack_shape_match
    unpack_shape_line = unpack_shape_match.group()
    shape_dim = [d.strip() for d in unpack_shape_line.split('=')[0].strip().split(',')]
    dim_index = {shape_dim[i] : i for i in range(len(shape_dim))}
    fused_dim = {}
    for axis in fused_axis_len_list:
        index = []
        for dim in fused_axis_len_list[axis]:
            index.append(dim_index[dim])
        fused_dim[axis] = index

    new_unpack_shape_str = indent + 'if len(shape) != ' + str(len(shape_dim)) + ':\n'
    for sd in shape_dim:
        new_unpack_shape_str += indent + indent + sd + ' = te.var(\'' + sd + '\')\n'
    new_unpack_shape_str += indent + 'else:\n' + indent + unpack_shape_line

    new_condition_return_line = indent + 'if for_rtile:\n'
    fused_all_axis = fused_axis[-1] + list(ra_fused_axis.keys())
    new_condition_return_line += indent + indent + ', '.join([a.upper() for a in fused_all_axis]) + ' = shape\n'
    new_condition_return_line += indent + indent + 'return ['
    for i in range(len(ins_name)):
        new_condition_return_line += '(\'' + ins_name[i] + '\', []), '
    for i in range(len(padded_ins_name)):
        new_condition_return_line += '(\'' + padded_ins_name[i] + '\', [' + ', '.join([a.upper() for a in fused_axis[i]]) + '])'
        if i == len(padded_ins_name) - 1:
            new_condition_return_line += '], ['
        else:
            new_condition_return_line += ', '
    new_condition_return_line += '(\'' + out_name + '\', [' + ', '.join([a.upper() for a in fused_axis[-1]]) + ']), '
    new_condition_return_line += '(\'' + final_out_name + '\', [' + ', '.join([a.upper() for a in fused_axis[-1]]) + '])]\n\n'

    new_unpack_shape_str = new_condition_return_line + new_unpack_shape_str

    return_line = 'return [' + ins_name[0]
    return_pos = expr_str.find(return_line)
    assert return_pos != -1

    all_ins = ins_name + padded_ins_name
    new_return_line = indent + 'return [' + ', '.join(all_ins) + '], [' + out_name + ', ' + final_out_name + '], ' + str(fused_dim) + '\n'

    # find if for_rtile return line
    for_rtile_return_match = 'return \[\((?s:.)+\]\)\]\n'
    x = re.search(for_rtile_return_match, expr_str)
    assert x
    for_rtile_return_line = x.group()

    # get added lines
    added_lines = raxis_fuse + '\n' + ins_pad + '\n' + new_out_expr + '\n' + final_out + '\n' + new_return_line

    comment_out_expr = indent + '\'\'\'\n' + out_expr  + '\n' + indent + '\'\'\'\n'
    new_expr_str = expr_str.replace('if for_rtile:', ins_pad_decl) # add input_pad_decl
    new_expr_str = new_expr_str.replace(out_expr, comment_out_expr) # comment original output compute expr
    new_expr_str = new_expr_str.replace(return_line, '#' + return_line) # comment original return line
    new_expr_str = new_expr_str.replace(for_rtile_return_line, '\n') # replace for rtile return line
    new_expr_str = new_expr_str.replace(unpack_shape_line, new_unpack_shape_str) # replace original unpack shape line with new one

    new_expr_str += added_lines

    # replace original expr name with new one
    ori_expr_start_pos = new_expr_str.find('def ')
    ori_expr_end_pos = new_expr_str.find('(')
    ori_expr_name = new_expr_str[ori_expr_start_pos + 4: ori_expr_end_pos]
    if 'conv_expr' in ori_expr_name:
        return globals()['fused_' + ori_expr_name]
    new_expr_str = new_expr_str.replace(ori_expr_name, new_expr_name, 1)
    print("############# new expr ####################")
    print(new_expr_str)
    print("###########################################")
    d = {}
    for m in globals():
        d[m] = globals()[m]
    exec(new_expr_str, d)

    return d[new_expr_name]