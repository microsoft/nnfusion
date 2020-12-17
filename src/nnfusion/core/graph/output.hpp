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

#include <memory>

#include "nnfusion/common/descriptor/tensor.hpp"

namespace nnfusion
{
    namespace graph
    {
        class Output
        {
        public:
            /// \param tensor The view of this tensor; where the value will be written
            Output(const std::shared_ptr<nnfusion::descriptor::Tensor>& tensor)
                : m_tensor(tensor)
            {
            }

            nnfusion::descriptor::Tensor& get_tensor() const { return *m_tensor; }
            std::shared_ptr<nnfusion::descriptor::Tensor> get_tensor_ptr() const
            {
                return m_tensor;
            }
            void set_tensor_ptr(const std::shared_ptr<nnfusion::descriptor::Tensor>& tensor)
            {
                m_tensor = tensor;
            }

            /// \return the element type of the output
            const nnfusion::element::Type& get_element_type() const
            {
                return m_tensor->get_element_type();
            }
            /// \return the shape of the output
            const nnfusion::Shape& get_shape() const { return m_tensor->get_shape(); }
            const nnfusion::PartialShape& get_partial_shape() const
            {
                return m_tensor->get_partial_shape();
            }

            void set_type_and_shape(const nnfusion::element::Type& element_type,
                                    const nnfusion::PartialShape& pshape)
            {
                m_tensor->set_tensor_type(element_type, pshape);
            }

        protected:
            std::shared_ptr<nnfusion::descriptor::Tensor> m_tensor;

        private:
            Output(const Output&) = delete;
            Output(Output&&) = delete;
            Output& operator=(const Output&) = delete;
        };
    }
}
