#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from ir_parser import ir_graph_parser


def get_input_dict(input_orders):
  input_list, input_dict = [], {}
  for k in input_orders:
    if isinstance(input_orders[k], tuple):
      input_list += [(k, input_orders[k][2], input_orders[k][1])]
    else:
      input_list += [(k, input_orders[k].shape, input_orders[k].dtype)]
  for k, shape, dtype in input_list:
    input_dict[k] = {
      'dtype': str(dtype).split('.')[1],
      'shape': list(shape)
    }
  for k in input_dict:
    if len(input_dict[k]['shape']) == 0:
      input_dict[k]['shape'] = [1] 
  return input_dict

def construct_json_graph(ir, input_dict):
  exprss = ir.replace('\n', ' ').strip()
  ast_seq, input_dict, output_dict, _ = ir_graph_parser(exprss, input_dict)
  # print('input_dict:', input_dict)
  # print('output_dict:', output_dict)
  # topological sort and construct graph
  nodes = []
  known_tensors = {k : v for v, k in enumerate(sorted(list(input_dict)))}
  node_index_offset = len(known_tensors)
  while not all([k in known_tensors for k in output_dict]):
    node_len = len(nodes)
    for index, ast in enumerate(ast_seq):
      node_output_name = ast['props']['output_name']
      if node_output_name in known_tensors:
        continue  # already added nodes
      node_input_list = list(ast['props']['input_dict'])
      node_input_list.sort(key=lambda x : ast['props']['raw_exprss'].find(x))
      if all([k in known_tensors for k in node_input_list]):
        # generate antares expression
        expression_ast = ast['props']['raw_exprss'].replace('"', '`').replace('\n', ' ').strip()
        input_dict_ast = json.dumps(ast['props']['input_dict'])
        expression_ast = f'- einstein_v2(" {expression_ast}", input_dict={input_dict_ast})'
        # replace input name in exprss
        # print('expression_ast old:', expression_ast)
        for v, k in enumerate(node_input_list):
          expression_ast = expression_ast.replace(k, 'input%d' % (v))
        # print('expression_ast new:', expression_ast)
        # count edges
        edges = [[known_tensors[k], 0] for k in node_input_list]
        # construct new node
        node_id = index + node_index_offset
        nodes.append([node_id, expression_ast, node_output_name, edges])
        known_tensors[node_output_name] = node_id
    if node_len == len(nodes) and not all([k in known_tensors for k in output_dict]):
      raise Exception('Invalid model graph.')
  # add output node
  node_index_offset += len(ast_seq)
  for v, k in enumerate(output_dict):
    nodes.append([v + node_index_offset, '', 'Result', [[known_tensors[k], 0]]])
  return json.dumps(nodes, indent=2)


if __name__ == '__main__':
  import torch
  # from antares_core.frameworks.pytorch.custom_op import CustomOp
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  dtype = torch.float32
  kwargs = {'dtype': dtype,
            'device': device,
            'requires_grad': False}
  def create_param(name, shape):
    return (torch.rand(shape, **kwargs) - 0.5) * 0.001
  input_tensor = torch.ones([64, 3, 227, 227], **kwargs)
  const_0_ = create_param('const_0_', [11, 11, 3, 64])
  const_1_ = create_param('const_1_', [5, 5, 64, 192])
  const_2_ = create_param('const_2_', [3, 3, 192, 384])
  const_3_ = create_param('const_3_', [3, 3, 384, 256])
  const_4_ = create_param('const_4_', [3, 3, 256, 256])
  const_5_ = create_param('const_5_', [9216, 4096])
  const_6_ = create_param('const_6_', [4096, 4096])
  const_7_ = create_param('const_7_', [4096, 1000])
  ir = f'''
    conv_0[N, F, HO, WO] +=! input_tensor[N, C, HO * 4 + KH, WO * 4 + KW] * const_0_[KH, KW, C, F] where HO in 55, WO in 55;
    mpool_0[N, C, HO, WO] >=! conv_0[N, C, HO * 2 + KH, WO * 2 + KW].call(`max`, [0.0]) where HO in 27, WO in 27, KH in 3, KW in 3;
    conv_1[N, F, HO, WO] +=! mpool_0[N, C, -2 + HO + KH, -2 + WO + KW].when([-2 + HO + KH >= 0, -2 + HO + KH < 27, -2 + WO + KW >= 0, -2 + WO + KW < 27], 0.0) * const_1_[KH, KW, C, F] where HO in 27, WO in 27;
    mpool_1[N, C, HO, WO] >=! conv_1[N, C, HO * 2 + KH, WO * 2 + KW].call(`max`, [0.0]) where HO in 13, WO in 13, KH in 3, KW in 3;
    conv_2[N, F, HO, WO] +=! mpool_1[N, C, -1 + HO + KH, -1 + WO + KW].when([-1 + HO + KH >= 0, -1 + HO + KH < 13, -1 + WO + KW >= 0, -1 + WO + KW < 13], 0.0) * const_2_[KH, KW, C, F] where HO in 13, WO in 13;
    conv_2_relu[N, F, HO, WO] = conv_2[N, F, HO, WO].call(`max`, [0.0]);
    conv_3[N, F, HO, WO] +=! conv_2_relu[N, C, -1 + HO + KH, -1 + WO + KW].when([-1 + HO + KH >= 0, -1 + HO + KH < 13, -1 + WO + KW >= 0, -1 + WO + KW < 13], 0.0) * const_3_[KH, KW, C, F] where HO in 13, WO in 13;
    conv_3_relu[N, F, HO, WO] = conv_3[N, F, HO, WO].call(`max`, [0.0]);
    conv_4[N, F, HO, WO] +=! conv_3_relu[N, C, -1 + HO + KH, -1 + WO + KW].when([-1 + HO + KH >= 0, -1 + HO + KH < 13, -1 + WO + KW >= 0, -1 + WO + KW < 13], 0.0) * const_4_[KH, KW, C, F] where HO in 13, WO in 13;
    mpool_2[N, C, HO, WO] >=! conv_4[N, C, HO * 2 + KH, WO * 2 + KW].call(`max`, [0.0]) where HO in 6, WO in 6, KH in 3, KW in 3;
    reshape_0[N0, N1] = mpool_2[N0, N1 // 36 % 256, N1 // 6 % 6, N1 % 6] where N1 in 9216;
    dense_0[N, M] +=! reshape_0[N, K] * const_5_[K, M];
    dense_0_relu[N, M] = dense_0[N, M].call(`max`, [0.0]);
    dense_1[N, M] +=! dense_0_relu[N, K] * const_6_[K, M];
    dense_1_relu[N, M] = dense_1[N, M].call(`max`, [0.0]);
    dense_2[N, M] +=! dense_1_relu[N, K] * const_7_[K, M];
  '''
  input_orders={
    'input_tensor': input_tensor,
    'const_0_': const_0_,
    'const_1_': const_1_,
    'const_2_': const_2_,
    'const_3_': const_3_,
    'const_4_': const_4_,
    'const_5_': const_5_,
    'const_6_': const_6_,
    'const_7_': const_7_,
  }
  input_dict = get_input_dict(input_orders)
  graph = construct_json_graph(ir, input_dict)
  with open('alexnet_ir_graph.json', 'w') as f:
    f.write(graph)

  # output_logits = CustomOp(ir, input_orders=input_orders, device=device).emit()
  # result = output_logits(input_tensor, const_0_, const_1_, const_2_, const_3_, const_4_, const_5_, const_6_, const_7_)
  # print('The result of tensor `%s` is:\n%s' % (output_logits.output_names[0], result))