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

#include "allreduce.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

AllReduce::AllReduce()
    : Op("AllReduce")
{
}

void AllReduce::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto element_type = gnode->get_input_element_type(0);
    OP_VALIDATION(this,
                  element_type.is_dynamic() || element_type == nnfusion::element::f32 ||
                      element_type == nnfusion::element::f64)
        << "Only element types f32 and f64 are supported (argument element type: " << element_type
        << ").";

    gnode->set_output_type_and_shape(0, element_type, gnode->get_input_partial_shape(0));
}