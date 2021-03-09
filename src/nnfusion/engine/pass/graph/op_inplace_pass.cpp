// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "op_inplace_pass.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/broadcast.hpp"
#include "nnfusion/core/operators/op_define/reduce.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/core/operators/op_define/result.hpp"
#include "nnfusion/core/operators/op_define/select.hpp"
#include "nnfusion/core/operators/util/arithmetic_reduction.hpp"
#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::op;
using namespace nnfusion::pass::graph;

bool OpInplacePass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    for (auto node : graph->get_nodes())
    {
        if (auto op = std::dynamic_pointer_cast<Result>(node->get_op_ptr()))
        {
            AddInplace(op, 0, 0, false);
        }

        else if (auto op = std::dynamic_pointer_cast<ElementwiseArithmetic>(node->get_op_ptr()))
        {
            AddInplace(op, 0, 0, true);
        }

        else if (auto op = std::dynamic_pointer_cast<Select>(node->get_op_ptr()))
        {
            AddInplace(op, 0, 1, true);
        }

        else if (node->get_op_type() == "AddN")
        {
            auto op = std::dynamic_pointer_cast<GenericOp>(node->get_op_ptr());
            AddInplace(op, 0, 0, true);
        }

        else if (node->get_op_type() == "ApplyGradient")
        {
            auto op = std::dynamic_pointer_cast<GenericOp>(node->get_op_ptr());
            AddInplace(op, 0, 0, true, true);
        }

        else if (node->get_op_type() == "ApplyGradientDescent")
        {
            auto op = std::dynamic_pointer_cast<GenericOp>(node->get_op_ptr());
            AddInplace(op, 0, 0, true, true);
        }

        else if (node->get_op_type() == "Reshape")
        {
            auto op = std::dynamic_pointer_cast<nnfusion::op::Reshape>(node->get_op_ptr());
            auto output_shape = op->get_output_shape();
            size_t output_size = shape_size(output_shape);
            if (!op->get_is_layout_change() || output_size < 2)
            {
                AddInplace(op, 0, 0, false);
            }
        }

        else if (node->get_op_type() == "Broadcast")
        {
            auto op = std::dynamic_pointer_cast<nnfusion::op::Broadcast>(node->get_op_ptr());
            auto& axes = op->get_broadcast_axes();
            auto arg_shape = node->get_input_shape(0);
            auto result_shape = node->get_output_shape(0);
            if (axes.empty() || shape_size(arg_shape) == shape_size(result_shape))
            {
                AddInplace(op, 0, 0, false);
            }
        }

        else if (node->get_op_type() == "AllReduce")
        {
            auto op = std::dynamic_pointer_cast<Op>(node->get_op_ptr());
            AddInplace(op, 0, 0, false);
        }

        else if (node->get_op_type() == "MatMulAdd")
        {
            auto op = std::dynamic_pointer_cast<GenericOp>(node->get_op_ptr());
            AddInplace(op, 0, 2, true);
        }

        else if (nnfusion::op::get_annotation(nnfusion::op::get_translation(node))
                     .find("|memcpy|") != string::npos)
        {
            auto op = node->get_op_ptr();
            AddInplace(op, 0, 0, false);
        }
    }
    return true;
}
