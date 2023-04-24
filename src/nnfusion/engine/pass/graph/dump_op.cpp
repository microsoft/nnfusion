// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "dump_op.hpp"

using namespace nnfusion;
using namespace nnfusion::pass::graph;
using namespace nnfusion::profiler;

DEFINE_string(fdump_op_file, "", "");

bool DumpOp::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (FLAGS_fdump_op_file == "")
        return true;

    std::ofstream out(FLAGS_fdump_op_file);
    out << "op\tinput\toutput\tfused nodes\n";
    std::vector<std::shared_ptr<GNode>> nodes = graph->get_ordered_ops();
    for (auto it : nodes)
    {
        out << it->get_op_type() << "\t";
        for (size_t i = 0; i < it->get_inputs().size(); i++)
        {
            auto in_shape = it->get_input_shape(i);
            out << in_shape;
        }
        out << "\t";

        for (size_t i = 0; i < it->get_outputs().size(); i++)
        {
            auto out_shape = it->get_output_shape(i);
            out << out_shape;
        }
        out << "\t";
        if (it->get_op_type() == "ElementWiseFused")
        {
            auto node = std::static_pointer_cast<FusedGNode>(it);
            NNFUSION_CHECK_NOT_NULLPTR(node);
            auto ctxs = node->get_op_contexts();
            for (auto c : ctxs)
            {
                out << c->op->get_op_type() << ", inputs: ";
                for (size_t j = 0; j < c->inputs.size(); j++)
                {
                    out << c->inputs[j]->get_shape() << ", ";
                }
                out << "outputs: ";
                for (size_t j = 0; j < c->outputs.size(); j++)
                {
                    out << c->outputs[j]->get_shape() << ", ";
                }
                out << "\t";
            }
        }
        else if (it->get_op_type() == "Convolution")
        {
            auto op = static_pointer_cast<nnfusion::op::Convolution>(it->get_op_ptr());
            NNFUSION_CHECK_NOT_NULLPTR(op);
            out << op->get_activation() << "\t";
        }
        out << "\n";
    }
    out << std::endl;

    return true;
}