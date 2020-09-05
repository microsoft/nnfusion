//  Copyright (c) Microsoft Corporation.
//  Licensed under the MIT License.

#pragma once

#include <queue>
#include <string>
#include <vector>

#include "../torchscript_base.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace torchscript_import
        {
            graph::GNodeVector
                convert_node(const TBlockPtr, NodeMap&, std::shared_ptr<nnfusion::graph::Graph>);
            graph::GNodeVector
                convert_block(const TBlockPtr, NodeMap&, std::shared_ptr<nnfusion::graph::Graph>);
            graph::GNodeVector convert_block(const TBlockPtr,
                                             const std::vector<at::Tensor>&,
                                             const std::vector<nnfusion::Shape>&,
                                             const std::vector<nnfusion::element::Type>&,
                                             NodeMap&,
                                             std::shared_ptr<nnfusion::graph::Graph>);

            class GraphConvert
            {
            public:
                GraphConvert(const std::shared_ptr<torch::jit::Graph> ts_graph)
                    : GraphConvert(ts_graph,
                                   std::vector<at::Tensor>(),
                                   std::vector<nnfusion::Shape>(),
                                   std::vector<nnfusion::element::Type>())
                {
                }

                GraphConvert(const std::shared_ptr<torch::jit::Graph> ts_graph,
                             const std::vector<at::Tensor>& weights,
                             const std::vector<nnfusion::Shape>& input_shapes,
                             const std::vector<nnfusion::element::Type>& input_types);

                std::shared_ptr<nnfusion::graph::Graph> get_graph() { return m_graph; }
            private:
                const std::shared_ptr<torch::jit::Graph> ts_graph_;
                std::shared_ptr<nnfusion::graph::Graph> m_graph;

                std::vector<at::Tensor> weights_;
                std::vector<nnfusion::Shape> input_shapes_;
                std::vector<nnfusion::element::Type> input_types_;
                NodeMap tnode2gnodes;
            };
        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace nnfusion
