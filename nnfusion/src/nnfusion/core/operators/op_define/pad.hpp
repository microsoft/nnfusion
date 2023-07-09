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
    namespace op
    {
        /// \brief Generic constant-padding operation.
        class Pad : public Op
        {
        public:
            /// \brief Constructs a generic padding operation.
            ///
            /// \param padding_below The padding-below widths.
            /// \param padding_above The padding-above widths.
            /// \param padding_interior The interior-padding widths.
            Pad(const nnfusion::Shape& padding_below,
                const nnfusion::Shape& padding_above,
                const nnfusion::Shape& padding_interior);

            /// \return The padding-below sizes.
            const nnfusion::Shape& get_padding_below() const { return m_padding_below; }
            /// \return The padding-above sizes.
            const nnfusion::Shape& get_padding_above() const { return m_padding_above; }
            /// \return The interior padding sizes.
            const nnfusion::Shape& get_padding_interior() const { return m_padding_interior; }
        protected:
            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            nnfusion::Shape m_padding_below;
            nnfusion::Shape m_padding_above;
            nnfusion::Shape m_padding_interior;
        };
    }
}
