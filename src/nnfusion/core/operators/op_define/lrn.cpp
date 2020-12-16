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

#include "lrn.hpp"
#include "nnfusion/core/graph/gedge.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace nnfusion::op;

LRN::LRN(double alpha, double beta, double bias, size_t nsize)
    : ElementwiseArithmetic("LRN")
    , m_alpha(alpha)
    , m_beta(beta)
    , m_bias(bias)
    , m_size(nsize)
{
}

void LRN::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    ElementwiseArithmetic::validate_and_infer_types(gnode);

    auto arg_gnode = gnode->get_in_edge(0)->get_src();
    OP_VALIDATION(this, arg_gnode->get_output_size() == 1)
        << "The input node of LRN must have exactly one output, while " << arg_gnode->get_op_type()
        << " has " << arg_gnode->get_output_size() << " outputs.";
    OP_VALIDATION(this, arg_gnode->get_output_shape(0).size() >= 3)
        << "The input node of LRN must have rank >= 3 (argument shape: "
        << arg_gnode->get_output_shape(0) << ").";
}
