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

//*****************************************************************************

#include <sstream>

#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/util/util.hpp"
#include "numpy_transpose.hpp"

#include "nnfusion/common/util.hpp"

namespace nnfusion
{
    namespace graph
    {
        std::string numpy_transpose_error_str(const nnfusion::AxisVector& order,
                                              const nnfusion::Shape& in_shape)
        {
            std::ostringstream os;
            os << "The axes order ";
            os << "[ " << nnfusion::join(order) << " ]";
            os << " is incompatible with the input shape ";
            os << "[ " << nnfusion::join(in_shape) << " ]";
            os << " during numpy_transpose.";
            return os.str();
        }

        std::shared_ptr<GNode> numpy_transpose(const std::shared_ptr<GNode>& gnode,
                                               nnfusion::AxisVector order,
                                               size_t output_index)
        {
            auto in_shape = gnode->get_output_shape(output_index);
            // default, reverse the order of the axes
            if (order.size() == 0)
            {
                auto n = in_shape.size();
                order = nnfusion::AxisVector(n);
                std::generate(order.begin(), order.end(), [&n]() { return --n; });
            }
            else if (order.size() == in_shape.size())
            {
                // validate that the axes order is valid, i.e., unique and the right size
                std::unordered_set<nnfusion::AxisVector::value_type> axes;
                for (auto o : order)
                {
                    if (o < in_shape.size() && !axes.count(o))
                    {
                        axes.insert(o);
                    }
                    else
                    {
                        NNFUSION_CHECK_FAIL() << numpy_transpose_error_str(order, in_shape);
                    }
                }
            }
            else
            {
                NNFUSION_CHECK_FAIL() << numpy_transpose_error_str(order, in_shape);
            }

            // create output shape
            nnfusion::Shape out_shape;
            for (size_t i = 0; i < in_shape.size(); ++i)
                out_shape.push_back(in_shape[order[i]]);

            // do the reshaping with the order
            auto reshape_op = std::make_shared<op::Reshape>(order, out_shape);
            auto reshape_gnode = std::make_shared<GNode>(
                reshape_op, GNodeIndexVector({GNodeIndex(gnode, output_index)}));
            reshape_op->revalidate_and_infer_types(reshape_gnode->shared_from_this());
            return reshape_gnode;
        }

    } // namespace builder
} // namespace ngraph
