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

#include "../op.hpp"

#include "nnfusion/common/axis_set.hpp"
#include "nnfusion/common/util.hpp"
namespace nnfusion
{
    namespace op
    {
        /// \brief Operation which "adds" axes to an input tensor, replicating elements from the input as needed along the new axes.
        class Broadcast : public Op
        {
        public:
            /// \brief Constructs a conversion operation.
            ///
            /// \param shape          The shape of the output tensor.
            /// \param broadcast_axes The axis positions (0-based) in the result that are being broadcast. The
            ///                        remaining axes in shape must be the same as the shape of arg.
            Broadcast(const nnfusion::Shape& shape, const nnfusion::AxisSet& broadcast_axes);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;
            void infer_shared_memory(std::shared_ptr<graph::GNode> gnode) override;

            /// \return A set containing the indices of the broadcast axes (0-based).
            const nnfusion::AxisSet& get_broadcast_axes() const { return m_broadcast_axes; }
            const nnfusion::Shape& get_broadcast_shape() const { return m_shape; }
            bool is_inner_broadcast() { return m_is_inner_broadcast; }
            bool is_outer_broadcast() { return m_is_outer_broadcast; }
            size_t get_inner_broadcast_size() { return m_inner_bc_size; }
            size_t get_outer_broadcast_size() { return m_outer_bc_size; }
        protected:
            Broadcast(const std::string& node_type,
                      const nnfusion::Shape& shape,
                      const nnfusion::AxisSet& broadcast_axes);

            virtual void infer_shape(std::shared_ptr<graph::GNode> gnode) {}
            void inner_or_outer_broadcast();
            nnfusion::Shape m_shape;
            nnfusion::AxisSet m_broadcast_axes;
            size_t m_inner_bc_size, m_outer_bc_size;
            bool m_is_inner_broadcast = false;
            bool m_is_outer_broadcast = false;
        };

        /// \brief Broadcast arg to the same shape as like_arg.
        class BroadcastLike : public Broadcast
        {
        public:
            /// \brief Broadcast arg to the same shape as like_arg.
            ///
            /// Once the shape of like_arg is known, this op will be replaced with an equivalent
            /// Broadcast op.
            ///
            /// \param like_arg Provides the shape for the result.
            /// \param broadcast_axes indicates which axes will be broadcast. If empty,
            /// arg must be scalar and all axes are broadcast.
            BroadcastLike(const nnfusion::AxisSet& broadcast_axes);

            void infer_shape(std::shared_ptr<graph::GNode> gnode) override;

        protected:
            nnfusion::AxisSet m_initial_broadcast_axes;
        };
    }
}
