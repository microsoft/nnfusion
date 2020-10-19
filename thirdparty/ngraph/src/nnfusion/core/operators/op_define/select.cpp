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

#include "select.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

Select::Select()
    : Op("Select")
{
}

void Select::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    OP_VALIDATION(this,
                  gnode->get_input_element_type(0).is_dynamic() ||
                      gnode->get_input_element_type(0) == element::boolean)
        << "Argument 0 does not have boolean element type (element type: "
        << gnode->get_input_element_type(0) << ").";

    nnfusion::PartialShape result_shape = gnode->get_input_partial_shape(0);

    OP_VALIDATION(
        this, nnfusion::PartialShape::merge_into(result_shape, gnode->get_input_partial_shape(1)))
        << "Argument shapes are inconsistent.";
    OP_VALIDATION(
        this, nnfusion::PartialShape::merge_into(result_shape, gnode->get_input_partial_shape(2)))
        << "Argument shapes are inconsistent.";

    nnfusion::element::Type result_et;

    OP_VALIDATION(this,
                  nnfusion::element::Type::merge(result_et,
                                                 gnode->get_input_element_type(1),
                                                 gnode->get_input_element_type(2)))
        << "Argument 1 and 2 element types are inconsistent.";

    gnode->set_output_type_and_shape(0, result_et, result_shape);
}