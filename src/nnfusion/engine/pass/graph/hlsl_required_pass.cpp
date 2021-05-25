// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "hlsl_required_pass.hpp"
#include "gnode_device_dispatcher.hpp"

DECLARE_string(fdefault_device);

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

namespace
{
    void replace_node(std::shared_ptr<GNode> old_node,
                      std::shared_ptr<GNode> new_node,
                      std::shared_ptr<nnfusion::graph::Graph>& graph)
    {
        graph->add_node(new_node);

        for (auto out_edge : old_node->get_out_edges())
        {
            auto dst_node = out_edge->get_dst();
            if (out_edge->is_control_edge())
            {
                graph->add_control_edge(new_node, dst_node);
            }
            else
            {
                auto new_input = make_shared<nnfusion::graph::Input>(new_node->get_element_type(),
                                                                     new_node->get_shape());
                dst_node->set_input(out_edge->get_dst_input(), new_input);
                graph->add_edge(new_node, 0, dst_node, out_edge->get_dst_input());
            }
        }
        graph->remove_node(old_node);
    }

    void revalidate_gnode(std::shared_ptr<GNode> node,
                          std::shared_ptr<nnfusion::graph::Graph>& graph)
    {
        for (auto in_edge : node->get_in_edges())
        {
            if (in_edge->is_control_edge())
            {
                continue;
            }
            auto src_node = in_edge->get_src();
            auto src_output = src_node->get_outputs()[in_edge->get_src_output()];
            auto cur_input = node->get_inputs()[in_edge->get_dst_input()];
            if (cur_input->get_element_type() != src_output->get_element_type() ||
                cur_input->get_shape() != src_output->get_shape())
            {
                auto new_input = make_shared<nnfusion::graph::Input>(src_output->get_element_type(),
                                                                     src_output->get_shape());
                node->set_input(in_edge->get_dst_input(), new_input);
            }
        }
        node->get_op_ptr()->revalidate_and_infer_types(node);
        node->get_op_ptr()->infer_shared_memory(node);
    }

    bool check_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
    {
        for (auto node : graph->get_nodes())
        {
            cout << "Check node: " << node->get_name() << endl;
            std::set<std::shared_ptr<nnfusion::graph::Edge>> in_edges;
            for (auto in_edge : node->get_in_edges())
            {
                if (!in_edge->is_control_edge())
                {
                    in_edges.insert(in_edge);
                    NNFUSION_CHECK(in_edge->get_src());
                    NNFUSION_CHECK(in_edge->get_src_output() < node->get_input_size());
                }
            }
            NNFUSION_CHECK(in_edges.size() == node->get_input_size());

            std::set<std::shared_ptr<nnfusion::graph::Edge>> out_edges;
            for (auto out_edge : node->get_out_edges())
            {
                if (!out_edge->is_control_edge())
                {
                    out_edges.insert(out_edge);
                    NNFUSION_CHECK(out_edge->get_dst());
                }
            }
        }
        return true;
    }
}

bool HLSLRequiredPass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (FLAGS_fdefault_device != "HLSL")
    {
        return true;
    }

    map<element::Type, element::Type> type_mapping = {{element::i64, element::i32},
                                                      {element::u64, element::u32}};

    //if (check_graph(graph)) cout << "Before conversion" << endl;
    auto gnodes = graph->get_ordered_ops();

    for (auto node : gnodes)
    {
        if (node->is_parameter())
        {
            auto iter = type_mapping.find(node->get_element_type());
            if (iter != type_mapping.end())
            {
                auto src_type = iter->first;
                auto dst_type = iter->second;
                NNFUSION_LOG(INFO) << "Convert Parameter " << node->get_name() << ": "
                                   << src_type.c_type_string() << " -> "
                                   << dst_type.c_type_string();
                auto old_op = std::dynamic_pointer_cast<op::Parameter>(node->get_op_ptr());
                NNFUSION_CHECK(old_op);
                auto new_op = std::make_shared<op::Parameter>(
                    dst_type, node->get_shape(), old_op->get_cacheable(), old_op->require_grad());
                new_op->set_name(node->get_name());

                std::shared_ptr<GNode> new_node =
                    graph->add_node_and_edge(new_op, GNodeIndexVector{});
                replace_node(node, new_node, graph);
                node = new_node;
            }
        }
        else if (node->is_constant())
        {
            auto iter = type_mapping.find(node->get_element_type());
            if (iter != type_mapping.end())
            {
                auto src_type = iter->first;
                auto dst_type = iter->second;
                NNFUSION_LOG(INFO) << "Convert Constant " << node->get_name() << ": "
                                   << src_type.c_type_string() << " -> "
                                   << dst_type.c_type_string();
                auto old_op = std::dynamic_pointer_cast<op::Constant>(node->get_op_ptr());
                NNFUSION_CHECK(old_op);
                auto data = old_op->get_value_strings();

                auto new_op = std::make_shared<op::Constant>(dst_type, node->get_shape(), data);
                new_op->set_name(node->get_name());

                std::shared_ptr<GNode> new_node =
                    graph->add_node_and_edge(new_op, GNodeIndexVector{});
                replace_node(node, new_node, graph);
                node = new_node;
            }
        }
        else if (node->get_op_type() == "Convert")
        {
            auto iter = type_mapping.find(node->get_element_type());
            if (iter != type_mapping.end())
            {
                auto src_type = iter->first;
                auto dst_type = iter->second;
                NNFUSION_LOG(INFO) << "Convert Convert " << node->get_name() << ": "
                                   << src_type.c_type_string() << " -> "
                                   << dst_type.c_type_string();
                auto old_op = std::dynamic_pointer_cast<op::Convert>(node->get_op_ptr());
                NNFUSION_CHECK(old_op);

                auto new_op = std::make_shared<op::Convert>(dst_type);
                new_op->set_name(node->get_name());

                std::shared_ptr<GNode> new_node =
                    graph->add_node_and_edge(new_op, {node->get_in_edge(0)->get_src()});
                replace_node(node, new_node, graph);
                node = new_node;
            }
        }
        revalidate_gnode(node, graph);
    }

    graph->set_default_parameters();
    //if (check_graph(graph)) cout << "After conversion" << endl;
    return true;
}
