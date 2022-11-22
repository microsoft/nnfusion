// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "loop.hpp"
#include "../util/graph_convert.hpp"
#include "../util/util.hpp"
#include "nnfusion/core/operators/op_define/loop.hpp"
#include "nnfusion/core/operators/op_define/while.hpp"
#include <map>

using namespace nnfusion::frontend::onnx_import;

/*
class Model(torch.jit.ScriptModule):
    def __init__(self):
        super(Model, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor, num_loop: int):
        ret = x
        for i in range(num_loop):
            ret = ret + x
        return ret

x = torch.ones([2, 2], dtype=torch.float32)
a = torch.tensor(5)


ir_version: 6
producer_name: "pytorch"
producer_version: "1.9"
graph {
  node {
    output: "2"
    name: "Constant_0"
    op_type: "Constant"
    attribute {
      name: "value"
      t {
        data_type: 9
        raw_data: "\001"
      }
      type: TENSOR
    }
  }
  node {
    input: "num_loop.1"
    input: "2"
    input: "x.1"
    output: "3"
    name: "Loop_1"
    op_type: "Loop"
    attribute {
      name: "body"
      g {
        node {
          input: "ret.9"
          input: "x.1"
          output: "7"
          name: "Add_2"
          op_type: "Add"
        }
        node {
          input: "2"
          output: "8"
          name: "Identity_3"
          op_type: "Identity"
        }
        name: "torch-jit-export1"
        input {
          name: "i"
          type {
            tensor_type {
              elem_type: 7
              shape {
              }
            }
          }
        }
        input {
          name: "cond"
          type {
            tensor_type {
              elem_type: 9
              shape {
              }
            }
          }
        }
        input {
          name: "ret.9"
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
        output {
          name: "8"
          type {
            tensor_type {
              elem_type: 9
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
    name: "num_loop.1"
    type {
      tensor_type {
        elem_type: 7
        shape {
        }
      }
    }
  }
  output {
    name: "3"
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
                NamedNodeVector TranslateLoopOp(const onnx::NodeProto& node_proto,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    NNFUSION_CHECK_FAIL()
                        << "This is a placeholder convert_func, please use the real one.";
                    return {};
                }

                NamedNodeVector TranslateLoopOp(
                    const onnx::NodeProto& node_proto,
                    const onnx::GraphProto& graph_proto,
                    const NodeMap& all_ng_nodes,
                    std::shared_ptr<nnfusion::graph::Graph> m_graph,
                    const std::unordered_map<std::string, ConvertFuncMap>& domain_convert_func_map,
                    const string& model_dir,
                    const std::unordered_map<std::string, std::int64_t>& domain2version,
                    const std::unordered_map<std::string, size_t>& dim_params)
                {
                    bool is_for_op = (node_proto.input(0) != "");

                    Node node(node_proto);
                    onnx::GraphProto loop_body_graph_proto =
                        node.get_attribute_value<onnx::GraphProto>("body");

                    std::unordered_map<std::string, int> node_inputs;
                    assert(loop_body_graph_proto.input_size() == node_proto.input_size());
                    int idx = 0;
                    if (is_for_op) {
                      for (const auto& input_proto : loop_body_graph_proto.input())
                      {
                          node_inputs[input_proto.name()] = idx++;
                          if (idx == 1)
                              node_inputs[input_proto.name()] = -1;
                      }
                      for (size_t i = 0; i < node_proto.input_size(); i++)
                      {
                          node_inputs[node_proto.input(i)] = i;
                      }
                    } else {
                      for (const auto& input_proto : loop_body_graph_proto.input())
                      {
                          node_inputs[input_proto.name()] = idx - 1; // iter_count is ignored
                          idx ++;
                      }
                      for (size_t i = 0; i < node_proto.input_size(); i++)
                      {
                          node_inputs[node_proto.input(i)] = i - 1;
                      }
                    }
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    // we need to know which graph output maps to which Loop op output
                    std::unordered_map<std::string, int> loop_output_map;
                    for (auto output : loop_body_graph_proto.output())
                    {
                        int idx = loop_output_map.size();
                        loop_output_map[output.name()] = idx;
                    }

                    // process loop_body_graph
                    std::shared_ptr<nnfusion::graph::Graph> loop_body_graph;
                    {
                        loop_body_graph_proto = complete_graphproto(loop_body_graph_proto, graph_proto);
                        GraphProtoConvert loop_body_graph_convert(loop_body_graph_proto,
                                                                  domain_convert_func_map,
                                                                  model_dir,
                                                                  domain2version,
                                                                  dim_params,
                                                                  all_ng_nodes,
                                                                  true);
                        loop_body_graph = loop_body_graph_convert.get_graph();
                    }
                    std::map<std::string, std::string> output_to_input;
                    std::map<std::string, std::vector<std::shared_ptr<nnfusion::graph::GNode>>> read_of_input;
                    std::set<std::string> output_name_set;
                    {
                        std::vector<std::string> input_names;
                        std::vector<std::string> output_names;
                        for (auto node: loop_body_graph->get_nodes()) {
                            if (node->get_op_type() == "Parameter") {
                                input_names.push_back(node->get_name());
                            }
                        }
                        for (auto node: loop_body_graph->get_outputs()) {
                            std::string name = node->get_name();
                            output_names.push_back(name);
                            output_name_set.insert(name);
                        }
                        if (is_for_op) {
                          for (size_t i = 0; i < output_names.size() - 1; i++) {
                              output_to_input[output_names[i]] = input_names[i + 2];
                          }
                        } else {
                          for (size_t i = 0; i < output_names.size(); i++) {
                              output_to_input[output_names[i]] = input_names[i + 1];
                          }
                        }
                    }
                    // write to the output node after read its corresponding node
                    // TOFIX: may cause circle dependency in graph
                    for (auto node: loop_body_graph->get_nodes()) {
                        for (auto edge: node->get_in_edges()) {
                            if (edge->get_src()->get_op_type() == "Parameter") {
                                read_of_input[edge->get_src()->get_name()].push_back(node);
                            }
                        }
                    }
                    for (auto node: loop_body_graph->get_nodes()) {
                        if (output_name_set.find(node->get_name()) != output_name_set.end()) {
                            for (auto read: read_of_input[output_to_input[node->get_name()]]) {
                                if (read != node) {
                                  loop_body_graph->add_control_edge(read, node);
                                  NNFUSION_LOG(INFO) << "add dependency " << read->get_name() << "|" << read->get_op_type() << " " << node->get_name() << "|" << node->get_op_type();
                                }
                            }
                        }
                    }
                    // hack for LSTM training forward: reshape(i)->add(i) reshape(i)->scatter(i) so add(i) should execute after scatter(i)
                    for (auto node: loop_body_graph->get_nodes()) {
                        if (node->get_op_type() == "Reshape") {
                            auto op = std::dynamic_pointer_cast<nnfusion::op::Reshape>(node->get_op_ptr());
                            if (!op->get_is_layout_change())
                            {
                                std::vector<std::shared_ptr<nnfusion::graph::GNode>> out_scatters;
                                std::vector<std::shared_ptr<nnfusion::graph::GNode>> out_adds;
                                for (auto edge: node->get_out_edges()) {
                                    auto out = edge->get_dst();
                                    if (out->get_op_type() == "ScatterND")
                                        out_scatters.push_back(out);
                                    if (edge->is_control_edge() && out->get_op_type() == "Add")
                                        out_adds.push_back(out);
                                }
                                for (auto scatter: out_scatters) {
                                    for (auto add: out_adds) {
                                        loop_body_graph->add_control_edge(scatter, add);
                                        NNFUSION_LOG(INFO) << "forward dependency " << scatter->get_name() << "|" << scatter->get_op_type() << " " << add->get_name() << "|" << add->get_op_type();
                                    }
                                }
                            }
                        }
                    }
                    std::vector<nnfusion::PartialShape> output_shapes;
                    std::vector<nnfusion::element::Type> output_types;
                    for (size_t i = 1; i < loop_body_graph_proto.output().size(); i++)
                    {
                        ValueInfo output_value_info(loop_body_graph_proto.output()[i], dim_params);
                        output_shapes.push_back(output_value_info.get_shape());
                        output_types.push_back(output_value_info.get_element_type());
                    }
                    for (auto node : loop_body_graph->get_ordered_ops())
                    {
                        if (node->get_op_type() == "Parameter")
                        {
                            auto item = node->get_name();
                            if (!node_inputs.count(item))
                            {
                                if (is_for_op) {
                                  node_inputs[item] = idx++;
                                } else {
                                  node_inputs[item] = idx - 1;
                                }
                                idx ++;
                                if (find_node_from_graph(m_graph, item) == nullptr)
                                {
                                    NNFUSION_CHECK(all_ng_nodes.count(item));
                                    auto node = all_ng_nodes.at(item)[0];
                                    NNFUSION_CHECK(node.gnode->get_op_type() == "Parameter" ||
                                                   node.gnode->get_op_type() == "Constant");
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

                    if (is_for_op) {
                      auto loop_op =
                          std::make_shared<op::Loop>(loop_body_graph, output_shapes, output_types);
                      loop_op->set_loop_output_map(loop_output_map);
                      loop_op->set_name(node_proto.name());
                      auto loop_gnode = m_graph->add_node_and_edge(
                          loop_op, input_indexes, /* output_size */ node_proto.output_size());

                      NamedNodeVector ret;
                      for (size_t i = 0; i < node_proto.output_size(); i++)
                      {
                          ret.push_back(NamedNode(node_proto.output(i), loop_gnode, i));
                      }
                      return ret;
                    } else {
                      auto while_op = std::make_shared<op::While>(loop_body_graph, output_shapes, output_types);
                      while_op->set_loop_output_map(loop_output_map);
                      while_op->set_name(node_proto.name());
                      input_indexes.erase(input_indexes.begin());
                      auto while_gnode = m_graph->add_node_and_edge(
                          while_op, input_indexes, /* output_size */ node_proto.output_size());
                      NamedNodeVector ret;
                      for (size_t i = 0; i < node_proto.output_size(); i++)
                      {
                          ret.push_back(NamedNode(node_proto.output(i), while_gnode, i));
                      }
                      return ret;
                    }

                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
