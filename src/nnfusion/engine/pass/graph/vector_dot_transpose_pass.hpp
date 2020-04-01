// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/core/operators/op_define/dot.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

DEFINE_bool(ftranspose_vecdot, false, "Enable vectdot transpose.");

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class VectorDotTransposePass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override
                {
                    bool using_pass = FLAGS_ftranspose_vecdot;
                    if (!using_pass)
                        return true;

                    LOG(INFO) << "Vector Dot Transpose Pass starts up for Graph: "
                              << graph->get_name();

                    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
                    std::set<std::shared_ptr<GNode>> const_nodes = {};
                    std::set<std::shared_ptr<GNode>> down_streams = {};

                    // Find nodes with all constant upstream nodes
                    for (auto& it : nodes)
                    {
                        if (it->get_op_type() == "Dot")
                        {
                            auto dot =
                                std::dynamic_pointer_cast<nnfusion::op::Dot>(it->get_op_ptr());
                            CHECK_NOT_NULLPTR(dot);
                            if (dot->get_transpose_B())
                                continue;
                            std::vector<std::shared_ptr<nnfusion::graph::Edge>> in_edges;
                            for (auto& edge : it->get_in_edges())
                            {
                                if (!edge->is_control_edge())
                                {
                                    in_edges.push_back(edge);
                                }
                            }

                            CHECK(in_edges.size() == 2);
                            auto input_0_gnode = it->get_in_edge(0)->get_src();
                            auto input_1_gnode = it->get_in_edge(1)->get_src();

                            auto p_const = std::dynamic_pointer_cast<nnfusion::op::Constant>(
                                input_1_gnode->get_op_ptr());
                            if (!input_1_gnode->is_constant() || p_const->is_parameter())
                                continue;
                            CHECK(input_0_gnode->get_output_size() == 1)
                                << input_0_gnode->get_op_type() << "must has exactly one output.";
                            auto input0_shape = input_0_gnode->get_output_shape(0);
                            if (input0_shape.size() != 2 || input0_shape[0] != 1)
                                continue;

                            auto output = input_1_gnode->get_outputs().at(0);
                            size_t dtype_size = output->get_element_type().size();
                            if (dtype_size != 4)
                                continue;
                            Shape new_shape = {output->get_shape()[1], output->get_shape()[0]};
                            std::vector<int> values(new_shape[0] * new_shape[1]);
                            for (int i = 0; i < new_shape[0]; ++i)
                                for (int j = 0; j < new_shape[1]; ++j)
                                    values[i * new_shape[1] + j] =
                                        ((int*)p_const->get_data_ptr())[i + j * new_shape[0]];

                            dot->get_transpose_B() = true;
                            CHECK(output->get_shape().size() == 2);
                            auto new_constant_op = std::make_shared<nnfusion::op::Constant>(
                                output->get_element_type(), new_shape, values.data());
                            auto new_constant_gnode =
                                std::make_shared<GNode>(new_constant_op, GNodeVector());
                            new_constant_op->revalidate_and_infer_types(
                                new_constant_gnode->shared_from_this());

                            graph->replace_node(input_1_gnode, new_constant_gnode);
                        }
                    }

                    LOG(INFO) << "";
                    LOG(INFO) << "Vector Dot Transpose Pass ends up for Graph: "
                              << graph->get_name();
                    LOG(INFO) << "";
                    return true;
                }
            };
        } // namespace pass
    }     // namespace graph
} // namespace nnfusion
