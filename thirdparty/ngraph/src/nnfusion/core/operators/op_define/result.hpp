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

namespace nnfusion
{
    namespace op
    {
        class Result : public Op
        {
        public:
            /// \brief Allows a value to be used as a graph result.
            Result();

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            virtual bool is_output() const override { return m_needs_copy_to_host; }
            void set_needs_default_layout(bool val) { m_needs_default_layout = val; }
            bool needs_default_layout() const { return m_needs_default_layout; }
            void set_needs_copy_to_host(bool val) { m_needs_copy_to_host = val; }
            bool needs_copy_to_host() const { return m_needs_copy_to_host; }
        private:
            bool m_needs_default_layout{false};
            bool m_needs_copy_to_host{true};
        };
    }
}
