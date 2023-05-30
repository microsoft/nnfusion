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

#include "quant_linear.hpp"
#include <cmath>
#include <limits>
#include "../util/util.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_7
            {
                NamedNodeVector
                    TranslateQuantLinearOp(const onnx::NodeProto& node_proto,
                                           const NodeMap& all_ng_nodes,
                                           std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    NNFUSION_LOG(INFO) << "Translating QuantLinear";

                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    auto A = input_indexes[0];
                    auto Qweight = input_indexes[1];
                    auto Scales = input_indexes[1];
                    auto Zeros = input_indexes[1];

                    Node node(node_proto);
                    nnfusion::op::OpConfig::any myConfig;
                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.name(), "QuantLinear", myConfig);
                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, input_indexes);
                    return NamedNodeVector{{node_proto.output(0), generic_gnode}};
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
