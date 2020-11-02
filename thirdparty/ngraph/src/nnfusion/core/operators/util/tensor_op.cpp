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

#include "tensor_op.hpp"
#include "nnfusion/core/graph/gnode.hpp"
using namespace std;
using namespace nnfusion::op;

TensorOp::TensorOp(const std::string& node_type,
                   const nnfusion::element::Type& element_type,
                   const nnfusion::Shape& shape)
    : Op(node_type)
    , m_shape(shape)
    , m_element_type(element_type)
{
}

void TensorOp::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    Op::validate_and_infer_types(gnode);

    gnode->set_output_type_and_shape(0, m_element_type, m_shape);
}
