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
                    const std::unordered_map<std::string, std::int64_t>& domain2version,
                    const std::unordered_map<std::string, size_t>& dim_params)
                {
                    Node node(node_proto);
                    onnx::GraphProto then_branch_graph_proto =
                        node.get_attribute_value<onnx::GraphProto>("then_branch");
                    onnx::GraphProto else_branch_graph_proto =
                        node.get_attribute_value<onnx::GraphProto>("else_branch");

                    std::unordered_map<std::string, int> node_inputs;
                    for (size_t i = 0; i < node_proto.input_size(); i++)
                    {
                        node_inputs[node_proto.input(i)] = i;
                    }
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);

                    // process then_branch graph and else_branch_graph
                    std::shared_ptr<nnfusion::graph::Graph> then_branch_graph;
                    std::shared_ptr<nnfusion::graph::Graph> else_branch_graph;
                    {
                        then_branch_graph_proto = complete_graphproto(then_branch_graph_proto);
                        GraphProtoConvert then_branch_graph_convert(then_branch_graph_proto,
                                                                    domain_convert_func_map,
                                                                    model_dir,
                                                                    domain2version,
                                                                    dim_params,
                                                                    all_ng_nodes,
                                                                    true);
                        then_branch_graph = then_branch_graph_convert.get_graph();
                        else_branch_graph_proto = complete_graphproto(else_branch_graph_proto);
                        GraphProtoConvert else_branch_graph_convert(else_branch_graph_proto,
                                                                    domain_convert_func_map,
                                                                    model_dir,
                                                                    domain2version,
                                                                    dim_params,
                                                                    all_ng_nodes,
                                                                    true);
                        else_branch_graph = else_branch_graph_convert.get_graph();
                    }
                    std::vector<nnfusion::PartialShape> output_shapes;
                    std::vector<nnfusion::element::Type> output_types;
                    for (size_t i = 0; i < then_branch_graph_proto.output().size(); i++)
                    {
                        ValueInfo output_value_info(then_branch_graph_proto.output()[i],
                                                    dim_params);
                        output_shapes.push_back(output_value_info.get_shape());
                        output_types.push_back(output_value_info.get_element_type());
                    }
                    std::unordered_map<std::string, int> output_map;
                    for (size_t i = 0; i < then_branch_graph_proto.output().size(); i++)
                        output_map[then_branch_graph_proto.output()[i].name()] = i;
                    for (size_t i = 0; i < then_branch_graph_proto.output().size(); i++)
                        output_map[else_branch_graph_proto.output()[i].name()] = i;

                    auto nodes = then_branch_graph->get_nodes();
                    for (auto node : else_branch_graph->get_nodes())
                        nodes.push_back(node);
                    for (auto node : nodes)
                    {
                        if (node->get_op_type() == "Parameter")
                        {
                            auto item = node->get_name();
                            if (!node_inputs.count(item))
                            {
                                int idx = node_inputs.size();
                                node_inputs[item] = idx;
                                if (find_node_from_graph(m_graph, item) == nullptr)
                                {
                                    NNFUSION_CHECK(all_ng_nodes.count(item));
                                    auto node = all_ng_nodes.at(item)[0];
                                    NNFUSION_CHECK(node.gnode->get_op_type() == "Parameter")
                                        << node.gnode->get_op_type();
                                    auto new_node = m_graph->add_node_and_edge(
                                        node.gnode->get_op_ptr(), graph::GNodeVector({}));
                                    input_indexes.push_back(GNodeIndex{new_node, 0});
                                }
                                else
                                {
                                    auto gnode = find_node_from_graph(m_graph, item);
                                    input_indexes.push_back(GNodeIndex{gnode, 0});
                                }
                            }
                            NNFUSION_CHECK(node_inputs.count(node->get_name()));
                            node->Set<int>("subgraph_input_map",
                                           int(node_inputs[node->get_name()]));
                        }
                    }
                    auto if_op = std::make_shared<op::If>(
                        then_branch_graph, else_branch_graph, output_shapes, output_types);
                    if_op->set_name(node_proto.name());
                    if_op->set_output_map(output_map);
                    auto if_gnode =
                        m_graph->add_node_and_edge(if_op, input_indexes, node_proto.output_size());
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
