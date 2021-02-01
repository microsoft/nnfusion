// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "core/node.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector
                    TranslateSkipLayerNormOp(const onnx::NodeProto& node_proto,
                                             const NodeMap& all_ng_nodes,
                                             std::shared_ptr<nnfusion::graph::Graph> m_graph);

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
