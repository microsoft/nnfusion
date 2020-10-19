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

#include "binary_elementwise_comparison.hpp"
#include "nnfusion/core/graph/gnode.hpp"

#include "nnfusion/common/partial_shape.hpp"
#include "nnfusion/common/type/element_type.hpp"

using namespace nnfusion::op;

BinaryElementwiseComparison::BinaryElementwiseComparison(const std::string& node_type)
    : Op(node_type)
{
}

void BinaryElementwiseComparison::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto args_et_pshape = validate_and_infer_elementwise_args(gnode);
    nnfusion::PartialShape& args_pshape = std::get<1>(args_et_pshape);

    gnode->set_output_type_and_shape(0, nnfusion::element::boolean, args_pshape);
}
