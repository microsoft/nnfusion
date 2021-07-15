// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "if.hpp"
#include "../util/graph_convert.hpp"
#include "../util/util.hpp"
#include "nnfusion/core/operators/op_define/if.hpp"

using namespace nnfusion::frontend::onnx_import;

/*
class Model(torch.jit.ScriptModule):
    def __init__(self):
        super(Model, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, cond: int):
        t = x - y
        if cond > 0:
            if cond > 1:
                return x * z
            else:
                return x * t
        else:
            return x + y

x = torch.ones([2, 2], dtype=torch.float32)
y = torch.ones([2, 2], dtype=torch.float32)
z = torch.ones([2, 2], dtype=torch.float32)


ir_version: 6
producer_name: "pytorch"
producer_version: "1.6"
graph {
  node {
    input: "x.1"
    input: "y.1"
    output: "4"
    name: "Sub_0"
    op_type: "Sub"
  }
  node {
    output: "5"
    name: "Constant_1"
    op_type: "Constant"
    attribute {
      name: "value"
      t {
        data_type: 7
        raw_data: "\000\000\000\000\000\000\000\000"
      }
      type: TENSOR
    }
  }
  node {
    input: "cond.1"
    input: "5"
    output: "6"
    name: "Greater_2"
    op_type: "Greater"
  }
  node {
    input: "6"
    output: "7"
    name: "If_3"
    op_type: "If"
    attribute {
      name: "then_branch"
      g {
        node {
          output: "8"
          name: "Constant_4"
          op_type: "Constant"
          attribute {
            name: "value"
            t {
              data_type: 7
              raw_data: "\001\000\000\000\000\000\000\000"
            }
            type: TENSOR
          }
        }
        node {
          input: "cond.1"
          input: "8"
          output: "9"
          name: "Greater_5"
          op_type: "Greater"
        }
        node {
          input: "9"
          output: "10"
          name: "If_6"
          op_type: "If"
          attribute {
            name: "then_branch"
            g {
              node {
                input: "x.1"
                input: "z.1"
                output: "11"
                name: "Mul_7"
                op_type: "Mul"
              }
              name: "torch-jit-export2"
              output {
                name: "11"
              }
            }
            type: GRAPH
          }
          attribute {
            name: "else_branch"
            g {
              node {
                input: "x.1"
                input: "4"
                output: "12"
                name: "Mul_8"
                op_type: "Mul"
              }
              name: "torch-jit-export3"
              output {
                name: "12"
              }
            }
            type: GRAPH
          }
        }
        name: "torch-jit-export1"
        output {
          name: "10"
        }
      }
      type: GRAPH
    }
    attribute {
      name: "else_branch"
      g {
        node {
          input: "x.1"
          input: "y.1"
          output: "13"
          name: "Add_9"
          op_type: "Add"
        }
        name: "torch-jit-export4"
        output {
          name: "13"
        }
      }
      type: GRAPH
    }
  }
  name: "torch-jit-export"
  input {
    name: "x.1"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 2
          }
        }
      }
    }
  }
  input {
    name: "y.1"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 2
          }
        }
      }
    }
  }
  input {
    name: "z.1"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 2
          }
        }
      }
    }
  }
  input {
    name: "cond.1"
    type {
      tensor_type {
        elem_type: 7
        shape {
        }
      }
    }
  }
  output {
    name: "7"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 2
          }
        }
      }
    }
  }
}
opset_import {
  version: 11
}
*/

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateIfOp(const onnx::NodeProto& node_proto,
                                              const NodeMap& all_ng_nodes,
                                              std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    NNFUSION_CHECK_FAIL()
                        << "This is a placeholder convert_func, please use the real one.";
                    return {};
                }

                NamedNodeVector TranslateIfOp(
                    const onnx::NodeProto& node_proto,
                    const NodeMap& all_ng_nodes,
                    std::shared_ptr<nnfusion::graph::Graph> m_graph,
                    const std::unordered_map<std::string, ConvertFuncMap>& domain_convert_func_map,
                    const string& model_dir,
                    const std::unordered_map<std::string, size_t>& dim_params)
                {
                    Node node(node_proto);
                    onnx::GraphProto then_branch_graph_proto =
                        node.get_attribute_value<onnx::GraphProto>("then_branch");
                    onnx::GraphProto else_branch_graph_proto =
                        node.get_attribute_value<onnx::GraphProto>("else_branch");

                    onnx::NodeProto completed_node_proto(node_proto);
                    auto then_branch_graph_inputs = extract_input(then_branch_graph_proto);
                    auto else_branch_graph_inputs = extract_input(else_branch_graph_proto);
                    std::unordered_set<std::string> node_inputs;
                    for (size_t i = 0; i < node_proto.input_size(); i++)
                    {
                        node_inputs.insert(node_proto.input(i));
                    }
                    for (auto item : then_branch_graph_inputs)
                    {
                        if (node_inputs.find(item) == node_inputs.end())
                        {
                            completed_node_proto.add_input(item);
                            node_inputs.insert(item);
                        }
                    }
                    for (auto item : else_branch_graph_inputs)
                    {
                        if (node_inputs.find(item) == node_inputs.end())
                        {
                            completed_node_proto.add_input(item);
                            node_inputs.insert(item);
                        }
                    }
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, completed_node_proto);

                    // process then_branch graph and else_branch_graph
                    std::shared_ptr<nnfusion::graph::Graph> then_branch_graph;
                    std::shared_ptr<nnfusion::graph::Graph> else_branch_graph;
                    {
                        then_branch_graph_proto = complete_graphproto(then_branch_graph_proto);
                        GraphProtoConvert then_branch_graph_convert(then_branch_graph_proto,
                                                                    domain_convert_func_map,
                                                                    model_dir,
                                                                    dim_params,
                                                                    all_ng_nodes,
                                                                    true);
                        then_branch_graph = then_branch_graph_convert.get_graph();

                        else_branch_graph_proto = complete_graphproto(else_branch_graph_proto);
                        GraphProtoConvert else_branch_graph_convert(else_branch_graph_proto,
                                                                    domain_convert_func_map,
                                                                    model_dir,
                                                                    dim_params,
                                                                    all_ng_nodes,
                                                                    true);
                        else_branch_graph = else_branch_graph_convert.get_graph();
                    }

                    auto if_op = std::make_shared<op::If>(then_branch_graph, else_branch_graph);
                    if_op->set_name(node_proto.name());
                    auto if_gnode = m_graph->add_node_and_edge(if_op, input_indexes);

                    NamedNodeVector ret;
                    for (size_t i = 0; i < node_proto.output_size(); i++)
                    {
                        ret.push_back(NamedNode(node_proto.output(i), if_gnode, i));
                    }

                    return ret;

                    // for (auto item : all_ng_nodes)
                    // {
                    //     std::cout << "NodeMap[" << item.first << "]: " << item.second.size() << std::endl;
                    // }

                    // // std::cout << then_branch_graph_proto.DebugString() << std::endl;
                    // // std::cout << else_branch_graph_proto.DebugString() << std::endl;

                    // std::vector<onnx::ValueInfoProto> model_inputs;
                    // for (auto i = 0; i < model_proto.graph().input_size(); i++)
                    // {
                    //     model_inputs.push_back(model_proto.graph().input(i));
                    // }

                    // for (auto i = 0; i < model_inputs.size(); i++)
                    // {
                    //     auto input = then_branch_graph_proto.add_input();
                    //     input->CopyFrom(model_inputs[i]);
                    // }
                    // for (auto i = 0; i < model_inputs.size(); i++)
                    // {
                    //     auto input = else_branch_graph_proto.add_input();
                    //     input->CopyFrom(model_inputs[i]);
                    // }

                    // onnx::ModelProto then_branch_model_proto = onnx::ModelProto(model_proto);
                    // then_branch_model_proto.set_allocated_graph(&then_branch_graph_proto);
                    // // GraphConvert then_branch_converter =
                    // //     GraphConvert(then_branch_model_proto, {}, "", all_ng_nodes);
                    // // auto then_branch_graph = then_branch_converter.get_graph();
                    // // auto then_branch_graph = std::make_shared()

                    // onnx::ModelProto else_branch_model_proto = onnx::ModelProto(model_proto);
                    // else_branch_model_proto.set_allocated_graph(&else_branch_graph_proto);
                    // // GraphConvert else_branch_converter =
                    // //     GraphConvert(else_branch_model_proto, {}, "", all_ng_nodes);
                    // // auto else_branch_graph = else_branch_converter.get_graph();

                    // // exit(1);
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
