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

#include <vector>

#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"
#include "scatter.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_11
            {
                NamedNodeVector
                    TranslateScatterNDOp(const onnx::NodeProto& node_proto,
                                         const NodeMap& all_ng_nodes,
                                         std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);

                    Node node(node_proto);

                    nnfusion::op::OpConfig::any myConfig;

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.output(0), "ScatterND", myConfig);
                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, input_indexes);

                    return {{node_proto.output(0), generic_gnode}};
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
