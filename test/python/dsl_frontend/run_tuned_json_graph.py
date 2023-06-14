import json
from ir_parser import ir_graph_parser
from kernel_packer import pack_kernel_slices, get_kernel_metadata

def get_device_source(raw_json_graph, tuned_json_graph, inputs_outputs_info=[]):
  def get_ir_from_einstein_v2(exprss):
    if exprss == '':
      return '', {}
    header = 'def einstein_v2(ir, input_dict, *args, **kwargs): return ir, input_dict\n'
    sandbox = dict()
    cmd = header + 'ir, input_dict = ' + exprss[2:]
    exec(cmd, sandbox)
    return sandbox['ir'], sandbox['input_dict']
  def parse_raw_graph(raw_json_graph):
    raw_graph_map = {node[0] : {'exprss' : node[1], 'op_type' : node[2], 'edges' : node[3], 'input_dict' : get_ir_from_einstein_v2(node[1])[1]} for node in raw_json_graph if node[2] != 'Result'}
    node_outputs = {k : None for k in raw_graph_map}
    const_id = 0
    global_inputs = []
    for node_id in raw_graph_map:
      for i, (src_id, src_idx) in enumerate(raw_graph_map[node_id]['edges']):
        assert src_idx == 0, 'currently only support single node output.'
        if src_id not in node_outputs:
          node_outputs[src_id] = {
            'name' : 'const_%d' % (const_id),
            'prop' : raw_graph_map[node_id]['input_dict']['input%d' % (i)]
          }
          global_inputs.append(src_id)
          const_id += 1
        elif node_outputs[src_id] is None:
          raise Exception('Invalid graph, the graph must be topo-sorted and does not have cycle.')
        else:
          node_outputs[src_id]['prop'] = raw_graph_map[node_id]['input_dict']['input%d' % (i)]
      node_outputs[node_id] = {
        'name' : raw_graph_map[node_id]['op_type'],
        'prop' : None
      }
    global_outputs = [k for k in node_outputs if node_outputs[k]['prop'] is None]
    # parse Antares expression to get output description
    for k in global_outputs:
      _, _, node_output_dict, _ = ir_graph_parser(*get_ir_from_einstein_v2(raw_graph_map[k]['exprss']))
      assert len(node_output_dict) == 1, 'currenlty only support single node output.'
      node_outputs[k]['prop'] = list(node_output_dict.values())[0]
    return raw_graph_map, node_outputs, global_inputs, global_outputs
  # parse raw json graph
  raw_graph_map, node_outputs, global_inputs, global_outputs = parse_raw_graph(raw_json_graph)
  # print('global inputs:', global_inputs)
  # print('global outputs:', global_outputs)
  # print('node outputs:')
  # for k in node_outputs:
  #   print('node:', k)
  #   print('output:', node_outputs[k])
  #   if k in raw_graph_map:
  #     print('node op type:', raw_graph_map[k]['op_type'])
  # parse tuned json graph
  tuned_json_graph.sort(key=lambda x : x['group_id'])
  kernel_slices = []
  irs = []
  for node_group in tuned_json_graph:
    node_input_list = [raw_graph_map[src_id]['edges'][src_idx][0] for src_id, src_idx in node_group['input_desc']]
    node_input_list = [(node_outputs[k]['name'], node_outputs[k]['prop']) for k in node_input_list]
    node_output_list = [(node_outputs[k]['name'], node_outputs[k]['prop']) for k, _ in node_group['output_desc']]
    thread_extent = {'block_size' : node_group['block_size'], 'grid_size' : node_group['grid_size']}
    kernel_slices.append((node_group['code'], node_group['name'], node_input_list, node_output_list, thread_extent))
    irs += [get_ir_from_einstein_v2(raw_graph_map[node_id]['exprss'])[0].strip() for node_id in node_group['nodes']]
    # print('name:', node_group['name'])
    # print('nodes:', node_group['nodes'])
    # print('input_desc:', node_group['input_desc'])
    # print('output_desc:', node_group['output_desc'])
    # print('node_input_list:', node_input_list)
    # print('node_output_list:', node_output_list)
  code = pack_kernel_slices(kernel_slices)
  global_input_dict = {node_outputs[k]['name'] : node_outputs[k]['prop'] for k in global_inputs}
  global_input_list = [(node_outputs[k]['name'], node_outputs[k]['prop']) for k in global_inputs]
  global_output_list = [(node_outputs[k]['name'], node_outputs[k]['prop']) for k in global_outputs]
  irs = ';   '.join(irs).replace('"', '`').replace('\n', ' ').strip()
  exprss = f'- einstein_v2(input_dict={json.dumps(global_input_dict)}, extra_outputs=[], exprss="{irs}")'
  metadata = get_kernel_metadata(exprss, global_input_list, global_output_list)
  device_source = '%s\n%s' % (metadata, code)
  inputs_outputs_info.append(global_input_list)
  inputs_outputs_info.append(global_output_list)
  return device_source


def run_custom_op():
  import torch
  from antares_core.frameworks.pytorch.custom_op import CustomOp
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
  output_logits = CustomOp(ir, input_orders=input_orders, device=device).emit()
  result = output_logits(input_tensor, const_0_, const_1_, const_2_, const_3_, const_4_, const_5_, const_6_, const_7_)
  return [input_tensor, const_0_, const_1_, const_2_, const_3_, const_4_, const_5_, const_6_, const_7_], result


if __name__ == '__main__':
  import importlib
  import torch
  # read model
  raw_model_path = 'data/alexnet_ir_graph.json'
  tuned_model_path = 'data/tuned.json'
  with open(raw_model_path) as f:
    raw_json_graph = json.load(f)
  with open(tuned_model_path) as f:
    tuned_json_graph = json.load(f)
  # prepare input dict
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  dtype = torch.float32
  kwargs = {'dtype': dtype,
            'device': device,
            'requires_grad': False}
  def create_param(name, shape):
    return (torch.rand(shape, **kwargs) - 0.5) * 0.001
  # get device source
  inputs_outputs_info = []
  device_source = get_device_source(raw_json_graph, tuned_json_graph, inputs_outputs_info)
  with open('kernel.cu', 'w') as f:
    f.write(device_source)
  # run model
  input_list, output_list = inputs_outputs_info
  args = [create_param(name, prop['shape']) for name, prop in input_list + output_list]
  custom_lib = importlib.import_module('antares_custom_torch_v2_c_cuda')  # must import torch first
  custom_key = custom_lib.inject(device_source)
  custom_lib.forward(custom_key, args)
  print(args[-1])
  # compare with Antares CustomOp
  args_ref, out_ref = run_custom_op()
  out = torch.empty_like(out_ref)
  custom_lib.forward(custom_key, args_ref + [out])
  print('error:', torch.sum(torch.abs(out - out_ref)))
