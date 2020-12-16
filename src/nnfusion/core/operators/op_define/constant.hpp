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

#include <cstring>
#include <sstream>

#include "../util/tensor_op.hpp"
#include "nnfusion/common/type/bfloat16.hpp"
#include "nnfusion/common/type/data_buffer.hpp"
#include "nnfusion/common/type/element_type.hpp"
#include "nnfusion/common/util.hpp"
#include "nnfusion/core/graph/gnode.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Class for constants.
        class Constant : public TensorOp
        {
        public:
            /// \brief Constructs a tensor constant.
            ///
            /// \param element_type The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param values A vector of literals for initializing the tensor constant. The size
            ///        of values must match the size of the shape.
            template <typename T>
            Constant(const nnfusion::element::Type& element_type,
                     nnfusion::Shape shape,
                     const std::vector<T>& values)
                : TensorOp("Constant", element_type, shape)
                , m_data(nnfusion::aligned_alloc(
                      m_element_type.size(), nnfusion::shape_size(m_shape) * m_element_type.size()))
            {
                OP_VALIDATION(this,
                              values.size() == 1 || values.size() == nnfusion::shape_size(m_shape))
                    << "Did not get the expected number of literals for a constant of shape "
                    << m_shape << " (got " << values.size() << ", expected "
                    << (nnfusion::shape_size(m_shape) == 1 ? "" : "1 or ")
                    << nnfusion::shape_size(m_shape) << ").";

                if (values.size() == 1)
                {
                    write_values(std::vector<T>(nnfusion::shape_size(m_shape), values[0]));
                }
                else
                {
                    write_values(values);
                }
            }

            /// \brief Constructs a tensor constant.
            ///
            /// \param element_type The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param values A DataBuffer of literals for initializing the tensor constant. The size
            ///        of values must match the size of the shape.
            Constant(const nnfusion::element::Type& element_type,
                     nnfusion::Shape shape,
                     const DataBuffer& values)
                : TensorOp("Constant", element_type, shape)
                , m_data(nnfusion::aligned_alloc(
                      m_element_type.size(), nnfusion::shape_size(m_shape) * m_element_type.size()))
            {
                OP_VALIDATION(this, values.size() == nnfusion::shape_size(m_shape))
                    << "Did not get the expected number of literals for a constant of shape "
                    << m_shape << " (got " << values.size() << ", expected "
                    << (nnfusion::shape_size(m_shape) == 1 ? "" : "1 or ")
                    << nnfusion::shape_size(m_shape) << ").";

                values.dump(m_data);
            }

            /// \brief Constructs a tensor constant
            ///        This constructor is mainly to support deserialization of constants.
            ///
            /// \param element_type The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param values A list of string values to use as the constant data.
            Constant(const nnfusion::element::Type& element_type,
                     nnfusion::Shape shape,
                     const std::vector<std::string>& values)
                : TensorOp("Constant", element_type, shape)
                , m_data(nnfusion::aligned_alloc(
                      m_element_type.size(), nnfusion::shape_size(m_shape) * m_element_type.size()))
            {
                OP_VALIDATION(this,
                              values.size() == 1 || values.size() == nnfusion::shape_size(m_shape))
                    << "Did not get the expected number of literals for a constant of shape "
                    << m_shape << " (got " << values.size() << ", expected "
                    << nnfusion::shape_size(m_shape) << ").";

                DataBuffer buf(element_type);
                size_t shape_size = nnfusion::shape_size(m_shape);

                buf.loadFromStrings(values, shape_size);

                buf.dump(m_data);
            }

            /// \brief Constructs a tensor constant with the same initialization value copied across
            //         the tensor. This constructor is to support deserialization of constants.
            ///
            /// \param element_type The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param data A void* to constant data.
            Constant(const nnfusion::element::Type& element_type,
                     const nnfusion::Shape& shape,
                     const void* data)
                : TensorOp("Constant", element_type, shape)
                , m_data(nullptr)
            {
                size_t size = nnfusion::shape_size(m_shape) * m_element_type.size();
                m_data = nnfusion::aligned_alloc(m_element_type.size(), size);
                std::memcpy(m_data, data, size);
            }

            virtual ~Constant() override;

            /// \return The initialization literals for the tensor constant.
            std::vector<std::string> get_value_strings() const;

            template <typename T>
            std::vector<T> get_vector() const
            {
                NNFUSION_CHECK(sizeof(T) <= m_element_type.size() ||
                               nnfusion::shape_size(m_shape) << 0)
                    << "Buffer over-read";

                std::vector<T> rc;
                const T* p = reinterpret_cast<const T*>(m_data);
                for (size_t i = 0; i < nnfusion::shape_size(m_shape); i++)
                {
                    rc.push_back(p[i]);
                }
                return rc;
            }

            DataBuffer get_buffer() const;
            const void* get_data_ptr() const { return m_data; }
            size_t get_data_size() const
            {
                return nnfusion::shape_size(m_shape) * m_element_type.size();
            }
            template <typename T>
            const T* get_data_ptr() const
            {
                return reinterpret_cast<T*>(m_data);
            }

            element::Type get_type() const { return m_element_type; }
            bool is_constant() const override { return true; }
            bool& is_weight() { return m_is_weight; }
        protected:
            template <typename T>
            void write_values(const std::vector<T>& values)
            {
                DataBuffer buf(m_element_type);
                buf.loadVector(values);
                buf.dump(m_data);
            }
            bool m_is_weight = false;
            void* m_data{nullptr};
            Constant(const Constant&) = delete;
            Constant(Constant&&) = delete;
            Constant operator=(const Constant*) = delete;
        };
    }
}
