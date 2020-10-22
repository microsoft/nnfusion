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

#include <memory>

#include "index_reduction.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

IndexReduction::IndexReduction(const std::string& node_type,
                               size_t axis,
                               const nnfusion::element::Type& index_element_type)
    : Op(node_type)
    , m_axis(axis)
    , m_index_element_type(index_element_type)
{
}

void IndexReduction::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    const nnfusion::PartialShape& arg_shape = gnode->get_input_partial_shape(0);
    nnfusion::Rank rank = arg_shape.rank();

    OP_VALIDATION(this, rank.is_dynamic() || size_t(rank) >= 1) << "Argument rank is zero.";
    OP_VALIDATION(this, rank.is_dynamic() || m_axis < size_t(rank))
        << "Reduction axis (" << m_axis << ") is not less than argument rank (" << rank << ").";
    OP_VALIDATION(this,
                  m_index_element_type == nnfusion::element::i32 ||
                      m_index_element_type == nnfusion::element::i64)
        << "Index element is neither i64 or i32.";

    nnfusion::PartialShape output_shape{nnfusion::PartialShape::dynamic()};

    if (!rank.is_dynamic())
    {
        std::vector<nnfusion::Dimension> output_dims(size_t(rank) - 1);
        size_t j = 0;

        for (size_t i = 0; i < size_t(rank) - 1; i++)
        {
            if (j == m_axis)
            {
                j++;
            }
            output_dims[i] = arg_shape[j++];
        }

        output_shape = nnfusion::PartialShape(output_dims);
    }

    gnode->set_output_type_and_shape(0, m_index_element_type, output_shape);
}
