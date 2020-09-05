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
                OP_VALIDATION(this, values.size() == nnfusion::shape_size(m_shape))
                    << "Did not get the expected number of literals for a constant of shape "
                    << m_shape << " (got " << values.size() << ", expected "
                    << nnfusion::shape_size(m_shape) << ".";

                // use double as intermedia type might produce unexpected overflow
                if (element_type == element::character)
                {
                    std::vector<char> dvalues = nnfusion::parse_string<char>(values);
                    if (values.size() == 1 && shape_size(m_shape) != 1)
                    {
                        dvalues = std::vector<char>(shape_size(m_shape), dvalues[0]);
                    }
                    write_values(dvalues);
                }
                else if (element_type.is_integral())
                {
                    if (element_type.is_signed())
                    {
                        std::vector<int64_t> dvalues = nnfusion::parse_string<int64_t>(values);
                        if (values.size() == 1 && shape_size(m_shape) != 1)
                        {
                            dvalues = std::vector<int64_t>(shape_size(m_shape), dvalues[0]);
                        }
                        write_values(dvalues);
                    }
                    else
                    {
                        std::vector<uint64_t> dvalues = nnfusion::parse_string<uint64_t>(values);
                        if (values.size() == 1 && shape_size(m_shape) != 1)
                        {
                            dvalues = std::vector<uint64_t>(shape_size(m_shape), dvalues[0]);
                        }
                        write_values(dvalues);
                    }
                }
                else
                {
                    std::vector<double> dvalues = nnfusion::parse_string<double>(values);
                    if (values.size() == 1 && shape_size(m_shape) != 1)
                    {
                        dvalues = std::vector<double>(shape_size(m_shape), dvalues[0]);
                    }
                    write_values(dvalues);
                }
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

            bool is_constant() const override { return true; }
            bool& is_weight() { return m_is_weight; }
        protected:
            template <typename T>
            void write_values(const std::vector<T>& values)
            {
                write_to_buffer(
                    m_element_type, m_shape, values, m_data, nnfusion::shape_size(m_shape));
            }

            template <typename T, typename U>
            void write_buffer(void* target, const std::vector<U>& source, size_t count)
            {
                T* p = reinterpret_cast<T*>(target);
                for (size_t i = 0; i < count; i++)
                {
                    p[i] = static_cast<T>(source[i]);
                }
            }

            template <typename T>
            void write_to_buffer(const nnfusion::element::Type& target_type,
                                 const nnfusion::Shape& target_shape,
                                 const std::vector<T>& source,
                                 void* target,
                                 size_t target_element_count)
            {
                NNFUSION_CHECK(source.size() == target_element_count)
                    << "Constant initializer does not match shape";

                if (target_type == nnfusion::element::boolean ||
                    target_type == nnfusion::element::character)
                {
                    write_buffer<char, T>(target, source, target_element_count);
                }
                else if (target_type == nnfusion::element::bf16)
                {
                    write_buffer<nnfusion::bfloat16, T>(target, source, target_element_count);
                }
                else if (target_type == nnfusion::element::f32)
                {
                    write_buffer<float, T>(target, source, target_element_count);
                }
                else if (target_type == nnfusion::element::f64)
                {
                    write_buffer<double, T>(target, source, target_element_count);
                }
                else if (target_type == nnfusion::element::i8)
                {
                    write_buffer<int8_t, T>(target, source, target_element_count);
                }
                else if (target_type == nnfusion::element::i16)
                {
                    write_buffer<int16_t, T>(target, source, target_element_count);
                }
                else if (target_type == nnfusion::element::i32)
                {
                    write_buffer<int32_t, T>(target, source, target_element_count);
                }
                else if (target_type == nnfusion::element::i64)
                {
                    write_buffer<int64_t, T>(target, source, target_element_count);
                }
                else if (target_type == nnfusion::element::u8)
                {
                    write_buffer<uint8_t, T>(target, source, target_element_count);
                }
                else if (target_type == nnfusion::element::u16)
                {
                    write_buffer<uint16_t, T>(target, source, target_element_count);
                }
                else if (target_type == nnfusion::element::u32)
                {
                    write_buffer<uint32_t, T>(target, source, target_element_count);
                }
                else if (target_type == nnfusion::element::u64)
                {
                    write_buffer<uint64_t, T>(target, source, target_element_count);
                }
                else
                {
                    NNFUSION_CHECK_FAIL() << "unsupported type";
                }
            }

            bool m_is_weight = false;
            void* m_data{nullptr};
            Constant(const Constant&) = delete;
            Constant(Constant&&) = delete;
            Constant operator=(const Constant*) = delete;
        };
    }
}
