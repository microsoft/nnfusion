// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "recursion.hpp"
#include <stack>
#include "../util/graph_convert.hpp"
#include "../util/util.hpp"
#include "nnfusion/core/operators/op_define/recursion.hpp"

using namespace nnfusion::frontend::onnx_import;

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector
                    TranslateRecursionOp(const onnx::NodeProto& node_proto,
                                         const NodeMap& all_ng_nodes,
                                         std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    NNFUSION_CHECK_FAIL()
                        << "This is a placeholder convert_func, please use the real one.";
                    return {};
                }

                NamedNodeVector TranslateRecursionOp(
                    const onnx::NodeProto& node_proto,
                    const NodeMap& all_ng_nodes,
                    std::shared_ptr<nnfusion::graph::Graph> m_graph,
                    const std::unordered_map<std::string, ConvertFuncMap>& domain_convert_func_map,
                    const string& model_dir,
                    const std::unordered_map<std::string, std::int64_t>& domain2version,
                    const std::unordered_map<std::string, size_t>& dim_params)
                {
                    static std::stack<std::vector<nnfusion::PartialShape>> output_shape_st;
                    static std::stack<std::vector<nnfusion::element::Type>> output_types_st;
                    if (node_proto.op_type() == "func_forward")
                    {
                        auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                        auto op = std::make_shared<op::FuncForward>();
                        op->set_name(node_proto.name());
                        auto gnode =
                            m_graph->add_node_and_edge(op, input_indexes, node_proto.output_size());
                        NamedNodeVector ret;
                        auto output_shapes = output_shape_st.top();
                        auto output_types = output_types_st.top();
                        for (size_t i = 0; i < node_proto.output_size(); i++)
                        {
                            ret.push_back(NamedNode(node_proto.output(i), gnode, i));
                            gnode->set_output_type_and_shape(i, output_types[i], output_shapes[i]);
                        }

                        return ret;
                    }
                    Node node(node_proto);
                    onnx::GraphProto body_graph_proto =
                        node.get_attribute_value<onnx::GraphProto>("body");
                    auto body_graph_inputs = extract_input(body_graph_proto);
                    std::unordered_map<std::string, int> node_inputs;
                    assert(body_graph_proto.input_size() == node_proto.input_size());
                    for (const auto& input_proto : body_graph_proto.input())
                    {
                        int input_idx = node_inputs.size();
                        node_inputs[input_proto.name()] = input_idx;
                    }
                    std::unordered_map<std::string, int> output_map;
                    for (auto output : body_graph_proto.output())
                    {
                        int idx = output_map.size();
                        output_map[output.name()] = idx;
                    }

                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    std::vector<nnfusion::PartialShape> output_shapes;
                    std::vector<nnfusion::element::Type> output_types;
                    for (size_t i = 0; i < body_graph_proto.output().size(); i++)
                    {
                        ValueInfo output_value_info(body_graph_proto.output()[i], dim_params);
                        output_shapes.push_back(output_value_info.get_shape());
                        output_types.push_back(output_value_info.get_element_type());
                    }
                    output_shape_st.push(output_shapes);
                    output_types_st.push(output_types);

                    // process loop_body_graph
                    std::shared_ptr<nnfusion::graph::Graph> body_graph;
                    {
                        body_graph_proto = complete_graphproto(body_graph_proto);
                        GraphProtoConvert body_graph_convert(body_graph_proto,
                                                             domain_convert_func_map,
                                                             model_dir,
                                                             domain2version,
                                                             dim_params,
                                                             all_ng_nodes,
                                                             true);
                        body_graph = body_graph_convert.get_graph();
                    }
                    output_shape_st.pop();
                    output_types_st.pop();
                    for (auto node : body_graph->get_ordered_ops())
                    {
                        if (node->get_op_type() == "Parameter")
                            node->Set<int>("subgraph_input_map",
                                           int(node_inputs[node->get_name()]));
                    }
                    auto recursion_op =
                        std::make_shared<op::Recursion>(body_graph, output_shapes, output_types);
                    recursion_op->set_name(node_proto.name());
                    recursion_op->set_output_map(output_map);
                    auto recursion_gnode = m_graph->add_node_and_edge(
                        recursion_op, input_indexes, /* output_size */ node_proto.output_size());

                    NamedNodeVector ret;
                    for (size_t i = 0; i < node_proto.output_size(); i++)
                    {
                        ret.push_back(NamedNode(node_proto.output(i), recursion_gnode, i));
                    }

                    return ret;
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
