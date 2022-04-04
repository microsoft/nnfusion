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

#include "result.hpp"
#include "nnfusion/core/graph/gedge.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

Result::Result()
    : Op("Result")
{
}

void Result::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    OP_VALIDATION(this, gnode->get_input_size() == 1)
        << "Argument has " << gnode->get_input_size() << " outputs (1 expected).";

    gnode->set_output_type_and_shape(
        0, gnode->get_input_element_type(0), gnode->get_input_partial_shape(0));
}
