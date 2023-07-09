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

#pragma once

#include "nnfusion/core/operators/op.hpp"

namespace nnfusion
{
    namespace graph
    {
        class Graph;
    }
    namespace op
    {
        /// \brief Windowed reduction operation.
        ///
        /// Slides a window of user-defined shape, with user-defined strides, over the tensor and produces for each window position the result obtained by
        /// reducing the tensors in the window to a scalar, using the user-supplied reduction graph.
        ///
        /// Given an input of shape \f$(d_1,\dots,d_n)\f$, a window shape of \f$(w_1,\dots,w_n)\f$ and window movement strides of \f$(s_1,\dots,s_n)\f$, the shape
        /// of the output is \f$(d'_1,\dots,d'_n)\f$ where \f$d'_i = \lceil \frac {d_i - w_i + 1}{s_i} \rceil\f$.
        ///
        /// ## Parameters
        ///
        /// |                           | Description                                                                                                               |
        /// | ------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
        /// | `reduction_graph`      | The scalar graph used to reduce the input tensor. Must take two arguments of type \f$E[]\f$ and return type \f$E[]\f$. |
        /// | `window_shape`            | The shape \f$(w_1,\dots,w_n)\f$ of the reduction window.                                                                  |
        /// | `window_movement_strides` | Movement strides \f$(s_1,\dots,s_n)\f$ to apply to the sliding window.                                                    |
        ///
        /// ## Inputs
        ///
        /// |                | Type                              | Description                                                                                           |
        /// | -------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------- |
        /// | `arg_reductee` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | An input tensor of any shape, with the element type matching that expected by the reduction graph. |
        /// | `arg_init`     | \f$E[]\f$                         | A scalar to be used as an initial value for reduction computations.                                   |
        ///
        /// ## Output
        ///
        /// | Type                     | Description                                                                                                                                                                                                                                                                                                                     |
        /// | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        /// | \f$E[d'_1,\dots,d'_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \mathit{reduce}(\mathit{reduction\_graph},\mathit{arg\_init},V)\f$ where \f$V\f$ is the set of values in the input tensor within the window defined by the lower bound \f$(s_1i_1,\dots,s_ni_n)\f$ and the noninclusive upper bound \f$(s_1i_1 + w_1,\dots,s_ni_n + w_n)\f$. |
        class ReduceWindow : public Op
        {
        public:
            /// \brief Constructs a reduce-window operation.
            ///
            /// \param reduction_graph The reduction graph to use.
            /// \param window_shape The window shape.
            /// \param window_movement_strides The window movement strides.
            ReduceWindow(const std::shared_ptr<graph::Graph>& reduction_graph,
                         const nnfusion::Shape& window_shape,
                         const nnfusion::Strides& window_movement_strides);

            /// \return A singleton vector containing the graph to use for reduction.
            std::vector<std::shared_ptr<graph::Graph>> get_graphs() const
            {
                return std::vector<std::shared_ptr<graph::Graph>>{m_reduction_graph};
            }
            /// \return The window shape.
            const nnfusion::Shape& get_window_shape() const { return m_window_shape; }
            /// \return The window movement strides.
            const nnfusion::Strides& get_window_movement_strides() const
            {
                return m_window_movement_strides;
            }

        protected:
            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            std::shared_ptr<graph::Graph> m_reduction_graph;
            nnfusion::Shape m_window_shape;
            nnfusion::Strides m_window_movement_strides;
        };
    }
}
