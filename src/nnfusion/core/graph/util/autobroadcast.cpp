//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

// Microsoft (c) 2019, NNFusion Team

#include "autobroadcast.hpp"

#include "nnfusion/core/operators/op_define/broadcast.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"

#include "nnfusion/common/axis_vector.hpp"
#include "nnfusion/common/util.hpp"

#include <cassert>
#include <memory>
#include <numeric>
#include <sstream>

using namespace std;

namespace nnfusion
{
    namespace graph
    {
        autobroadcast_incompatible_shapes::autobroadcast_incompatible_shapes(
            const nnfusion::Shape& shape1, const nnfusion::Shape& shape2)
            : errors::CheckError(error_str(shape1, shape2))
            , m_shape1(shape1)
            , m_shape2(shape2)
        {
        }

        const nnfusion::Shape& autobroadcast_incompatible_shapes::get_shape1() const
        {
            return m_shape1;
        }

        const nnfusion::Shape& autobroadcast_incompatible_shapes::get_shape2() const
        {
            return m_shape2;
        }

        std::string autobroadcast_incompatible_shapes::error_str(const nnfusion::Shape& shape1,
                                                                 const nnfusion::Shape& shape2)
        {
            ostringstream os;
            os << "Auto-broadcast not possible for these input shapes:"
               << " shape1=" << vector_to_string(shape1) << " shape2=" << vector_to_string(shape2);
            return os.str();
        }

        /// A utility struct representing the details computed by the
        /// compute_shapes_and_broadcast_axes function.
        struct Autobroadcast_plan
        {
            nnfusion::Shape m_arg1_shape_after_possible_reshaping;
            nnfusion::Shape m_arg2_shape_after_possible_reshaping;
            nnfusion::AxisSet m_arg1_broadcast_axes;
            nnfusion::AxisSet m_arg2_broadcast_axes;
            nnfusion::Shape m_final_shape;
        };

        /// \brief Compute the details regarding what reshape and/or broadcast operations must be applied to
        /// arg1 and/or arg2, as well as what the final resulting shape shall be.
        ///
        /// If this algorithm cannot handle the particular combination of shapes supplied as inputs, throw
        /// an nnfusion::builder::autobroadcast_incompatible_shapes exception.
        ///
        /// \exception nnfusion::builder::autobroadcast_incompatible_shapes
        static Autobroadcast_plan
            compute_shapes_and_broadcast_axes(const nnfusion::Shape& arg1_in_shape,
                                              const nnfusion::Shape& arg2_in_shape)
        {
            Autobroadcast_plan plan;

            size_t arg1_size = arg1_in_shape.size();
            size_t arg2_size = arg2_in_shape.size();
            size_t axis = std::max(arg1_size, arg2_size) - 1;

            // per numpy definition of broadcast:
            // start with trailing dimensions and work forward
            // two dimensions are compatible:
            //  * if they are equal
            //  * if one of them is 1
            while (arg1_size >= 1 || arg2_size >= 1)
            {
                size_t arg1_dim = arg1_size ? arg1_in_shape[arg1_size - 1] : 1;
                size_t arg2_dim = arg2_size ? arg2_in_shape[arg2_size - 1] : 1;

                if (arg1_dim == arg2_dim)
                {
                    // add dimension to broadcast shape + arg1/arg2 reshape
                    plan.m_final_shape.insert(plan.m_final_shape.begin(), arg1_dim);
                    plan.m_arg1_shape_after_possible_reshaping.insert(
                        plan.m_arg1_shape_after_possible_reshaping.begin(), arg1_dim);
                    plan.m_arg2_shape_after_possible_reshaping.insert(
                        plan.m_arg2_shape_after_possible_reshaping.begin(), arg2_dim);
                }
                else if (arg2_dim == 1)
                {
                    // add arg1 dimension to broadcast shape and arg1 reshape
                    plan.m_final_shape.insert(plan.m_final_shape.begin(), arg1_dim);
                    plan.m_arg1_shape_after_possible_reshaping.insert(
                        plan.m_arg1_shape_after_possible_reshaping.begin(), arg1_dim);
                    // add current axis to arg2 broadcast axes
                    plan.m_arg2_broadcast_axes.insert(plan.m_arg2_broadcast_axes.begin(), axis);
                }
                else if (arg1_dim == 1)
                {
                    // add arg2 dimension to broadcast shape and arg2 reshape
                    plan.m_final_shape.insert(plan.m_final_shape.begin(), arg2_dim);
                    plan.m_arg2_shape_after_possible_reshaping.insert(
                        plan.m_arg2_shape_after_possible_reshaping.begin(), arg2_dim);
                    // add current axis to arg1 broadcast axes
                    plan.m_arg1_broadcast_axes.insert(plan.m_arg1_broadcast_axes.begin(), axis);
                }
                else
                {
                    throw autobroadcast_incompatible_shapes(arg1_in_shape, arg2_in_shape);
                }

                if (arg1_size)
                {
                    --arg1_size;
                }

                if (arg2_size)
                {
                    --arg2_size;
                }

                if (axis)
                {
                    --axis;
                }
            }

            return plan;
        }

        /// If necessary, wrap \p node with an additional reshape and/or broadcast op.
        /// Return a pointer to the node that produces the wrapped value.
        /// If no additional reshape or broadcast op was needed, simply return \p node.
        static std::shared_ptr<GNode>
            add_required_ops(const std::shared_ptr<GNode>& gnode,
                             const nnfusion::Shape& node_shape_after_possible_reshaping,
                             const nnfusion::AxisSet& node_broadcast_axes,
                             const nnfusion::Shape& node_final_shape,
                             std::shared_ptr<nnfusion::graph::Graph> graph)
        {
            std::shared_ptr<GNode> return_gnode = gnode;

            if (gnode->get_shape() != node_shape_after_possible_reshaping)
            {
                // tell reshape to examine input dimensions in order
                nnfusion::AxisVector order = nnfusion::get_default_order(gnode->get_shape());
                auto return_op =
                    std::make_shared<op::Reshape>(order, node_shape_after_possible_reshaping);
                return_gnode = graph->add_node_and_edge(return_op, {return_gnode});
            }

            if (node_final_shape != node_shape_after_possible_reshaping)
            {
                auto return_op =
                    std::make_shared<op::Broadcast>(node_final_shape, node_broadcast_axes);
                return_gnode = graph->add_node_and_edge(return_op, {return_gnode});
            }

            return return_gnode;
        }

        static GNodeIndex
            add_required_ops(const GNodeIndex& gnode_index,
                             const nnfusion::Shape& node_shape_after_possible_reshaping,
                             const nnfusion::AxisSet& node_broadcast_axes,
                             const nnfusion::Shape& node_final_shape,
                             std::shared_ptr<nnfusion::graph::Graph> graph)
        {
            auto gnode = gnode_index.gnode;
            auto return_gnode_index = gnode_index;

            if (gnode->get_output_shape(gnode_index.index) != node_shape_after_possible_reshaping)
            {
                // tell reshape to examine input dimensions in order
                nnfusion::AxisVector order =
                    nnfusion::get_default_order(gnode->get_output_shape(gnode_index.index));
                auto return_op =
                    std::make_shared<op::Reshape>(order, node_shape_after_possible_reshaping);
                return_gnode_index =
                    GNodeIndex{graph->add_node_and_edge(return_op, {return_gnode_index})};
            }

            if (node_final_shape != node_shape_after_possible_reshaping)
            {
                auto return_op =
                    std::make_shared<op::Broadcast>(node_final_shape, node_broadcast_axes);
                return_gnode_index =
                    GNodeIndex{graph->add_node_and_edge(return_op, {return_gnode_index})};
            }

            return return_gnode_index;
        }

        std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>
            numpy_broadcast(const std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>& args,
                            std::shared_ptr<nnfusion::graph::Graph> graph)
        {
            assert(args.first);
            assert(args.second);

            const nnfusion::Shape& arg1_in_shape = args.first->get_shape();
            const nnfusion::Shape& arg2_in_shape = args.second->get_shape();

            // Handle the trivial case...
            if (arg1_in_shape == arg2_in_shape)
            {
                return args;
            }

            Autobroadcast_plan plan =
                compute_shapes_and_broadcast_axes(arg1_in_shape, arg2_in_shape);

            auto arg1_out = add_required_ops(args.first,
                                             plan.m_arg1_shape_after_possible_reshaping,
                                             plan.m_arg1_broadcast_axes,
                                             plan.m_final_shape,
                                             graph);

            auto arg2_out = add_required_ops(args.second,
                                             plan.m_arg2_shape_after_possible_reshaping,
                                             plan.m_arg2_broadcast_axes,
                                             plan.m_final_shape,
                                             graph);

            return {arg1_out, arg2_out};
        }

        std::pair<GNodeIndex, GNodeIndex>
            numpy_broadcast(const std::pair<GNodeIndex, GNodeIndex>& args,
                            std::shared_ptr<nnfusion::graph::Graph> graph)
        {
            assert(args.first.gnode);
            assert(args.second.gnode);

            const nnfusion::Shape& arg1_in_shape =
                args.first.gnode->get_output_shape(args.first.index);
            const nnfusion::Shape& arg2_in_shape =
                args.second.gnode->get_output_shape(args.second.index);

            // Handle the trivial case...
            if (arg1_in_shape == arg2_in_shape)
            {
                return args;
            }

            Autobroadcast_plan plan =
                compute_shapes_and_broadcast_axes(arg1_in_shape, arg2_in_shape);

            auto arg1_out = add_required_ops(args.first,
                                             plan.m_arg1_shape_after_possible_reshaping,
                                             plan.m_arg1_broadcast_axes,
                                             plan.m_final_shape,
                                             graph);

            auto arg2_out = add_required_ops(args.second,
                                             plan.m_arg2_shape_after_possible_reshaping,
                                             plan.m_arg2_broadcast_axes,
                                             plan.m_final_shape,
                                             graph);

            return {arg1_out, arg2_out};
        }

    } // namespace builder
} // namespace nnfusion
