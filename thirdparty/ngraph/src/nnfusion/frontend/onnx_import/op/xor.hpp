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

//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "ngraph/node_vector.hpp"
#include "nnfusion/core/operators/and.hpp"
#include "nnfusion/core/operators/not.hpp"
#include "nnfusion/core/operators/or.hpp"

#include "core/node.hpp"
#include "utils/broadcasting.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline NodeVector logical_xor(const Node& node)
                {
                    NodeVector ng_inputs{
                        numpy_style_broadcast_for_binary_operation(node.get_ng_inputs())};
                    auto left = ng_inputs.at(0);
                    auto not_left = std::make_shared<ngraph::op::Not>(left);
                    auto right = ng_inputs.at(1);
                    auto not_right = std::make_shared<ngraph::op::Not>(right);
                    return {std::make_shared<ngraph::op::Or>(
                        std::make_shared<ngraph::op::And>(left, not_right),
                        std::make_shared<ngraph::op::And>(not_left, right))};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
