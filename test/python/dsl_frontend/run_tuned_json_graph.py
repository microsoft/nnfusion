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
