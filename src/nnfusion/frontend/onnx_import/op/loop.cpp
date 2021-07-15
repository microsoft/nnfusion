// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "loop.hpp"
#include "../util/graph_convert.hpp"
#include "../util/util.hpp"
#include "nnfusion/core/operators/op_define/loop.hpp"

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
                    const NodeMap& all_ng_nodes,
                    std::shared_ptr<nnfusion::graph::Graph> m_graph,
                    const std::unordered_map<std::string, ConvertFuncMap>& domain_convert_func_map,
                    const string& model_dir,
                    const std::unordered_map<std::string, size_t>& dim_params)
                {
                    Node node(node_proto);
                    onnx::GraphProto loop_body_graph_proto =
                        node.get_attribute_value<onnx::GraphProto>("body");

                    onnx::NodeProto completed_node_proto(node_proto);
                    auto loop_body_graph_inputs = extract_input(loop_body_graph_proto);
                    std::unordered_set<std::string> node_inputs;
                    for (size_t i = 0; i < node_proto.input_size(); i++)
                    {
                        node_inputs.insert(node_proto.input(i));
                    }
                    for (const auto& input_proto : loop_body_graph_proto.input())
                    {
                        node_inputs.insert(input_proto.name());
                    }
                    for (auto item : loop_body_graph_inputs)
                    {
                        if (node_inputs.find(item) == node_inputs.end())
                        {
                            completed_node_proto.add_input(item);
                            node_inputs.insert(item);
                        }
                    }

                    auto input_indexes = GetAllInputIndex(all_ng_nodes, completed_node_proto);

                    // process loop_body_graph
                    std::shared_ptr<nnfusion::graph::Graph> loop_body_graph;
                    {
                        loop_body_graph_proto = complete_graphproto(loop_body_graph_proto);
                        std::cout << loop_body_graph_proto.DebugString() << std::endl;
                        GraphProtoConvert loop_body_graph_convert(loop_body_graph_proto,
                                                                  domain_convert_func_map,
                                                                  model_dir,
                                                                  dim_params,
                                                                  all_ng_nodes,
                                                                  true);
                        loop_body_graph = loop_body_graph_convert.get_graph();
                    }

                    std::vector<nnfusion::PartialShape> output_shapes;
                    std::vector<nnfusion::element::Type> output_types;
                    for (size_t i = 1; i < loop_body_graph_proto.output().size(); i++)
                    {
                        ValueInfo output_value_info(loop_body_graph_proto.output()[i], dim_params);
                        output_shapes.push_back(output_value_info.get_shape());
                        output_types.push_back(output_value_info.get_element_type());
                    }

                    auto loop_op =
                        std::make_shared<op::Loop>(loop_body_graph, output_shapes, output_types);
                    loop_op->set_name(node_proto.name());
                    auto loop_gnode = m_graph->add_node_and_edge(loop_op, input_indexes);

                    NamedNodeVector ret;
                    for (size_t i = 0; i < node_proto.output_size(); i++)
                    {
                        ret.push_back(NamedNode(node_proto.output(i), loop_gnode, i));
                    }

                    return ret;
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
