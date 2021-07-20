/*
ir_version: 6
producer_name: "pytorch"
producer_version: "1.6"
graph {
  node {
    output: "1"
    name: "Constant_0"
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
    output: "2"
    name: "Constant_1"
    op_type: "Constant"
    attribute {
      name: "value"
      t {
        dims: 2
        dims: 2
        data_type: 1
        raw_data: "\000\000\200?\000\000\200?\000\000\200?\000\000\200?"
      }
      type: TENSOR
    }
  }
  node {
    input: "1"
    output: "3"
    name: "Cast_2"
    op_type: "Cast"
    attribute {
      name: "to"
      i: 9
      type: INT
    }
  }
  node {
    input: "num_loop.1"
    input: "3"
    input: "2"
    output: "4"
    name: "Loop_3"
    op_type: "Loop"
    attribute {
      name: "body"
      g {
        node {
          output: "8"
          name: "Constant_4"
          op_type: "Constant"
          attribute {
            name: "value"
            t {
              dims: 2
              dims: 2
              data_type: 1
              raw_data: "\000\000\200?\000\000\200?\000\000\200?\000\000\200?"
            }
            type: TENSOR
          }
        }
        node {
          input: "ret.6"
          input: "8"
          output: "9"
          name: "Add_5"
          op_type: "Add"
        }
        node {
          input: "1"
          output: "10"
          name: "Cast_6"
          op_type: "Cast"
          attribute {
            name: "to"
            i: 9
            type: INT
          }
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
            }
          }
        }
        input {
          name: "ret.6"
        }
        output {
          name: "10"
        }
        output {
          name: "9"
        }
      }
      type: GRAPH
    }
  }
  name: "torch-jit-export"
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
    name: "4"
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

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "core/node.hpp"

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
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph);

                NamedNodeVector TranslateLoopOp(
                    const onnx::NodeProto& node_proto,
                    const NodeMap& all_ng_nodes,
                    std::shared_ptr<nnfusion::graph::Graph> m_graph,
                    const std::unordered_map<std::string, ConvertFuncMap>& domain_convert_func_map,
                    const string& model_dir,
                    const std::unordered_map<std::string, size_t>& dim_params = {});

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
