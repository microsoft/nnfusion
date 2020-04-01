// Microsoft (c) 2019, NNFusion Team

#pragma once

#include <cstring>
#include <sstream>

#include "nnfusion/common/type/bfloat16.hpp"
#include "nnfusion/common/type/element_type.hpp"
#include "nnfusion/common/util.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/operators/op.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Class for constants.
        class Constant : public Op
        {
        public:
            /// \brief Constructs a tensor constant.
            ///
            /// \param type The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param values A vector of literals for initializing the tensor constant. The size
            ///        of values must match the size of the shape.
            template <typename T>
            Constant(const nnfusion::element::Type& type,
                     nnfusion::Shape shape,
                     const std::vector<T>& values)
                : Op("Constant")
                , m_element_type(type)
                , m_shape(shape)
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
            /// \param type The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param values A list of string values to use as the constant data.
            Constant(const nnfusion::element::Type& type,
                     nnfusion::Shape shape,
                     const std::vector<std::string>& values)
                : Op("Constant")
                , m_element_type(type)
                , m_shape(shape)
                , m_data(nnfusion::aligned_alloc(
                      m_element_type.size(), nnfusion::shape_size(m_shape) * m_element_type.size()))
            {
                OP_VALIDATION(this, values.size() == nnfusion::shape_size(m_shape))
                    << "Did not get the expected number of literals for a constant of shape "
                    << m_shape << " (got " << values.size() << ", expected "
                    << nnfusion::shape_size(m_shape) << ".";

                std::vector<double> dvalues = nnfusion::parse_string<double>(values);
                write_values(dvalues);
            }

            /// \brief Constructs a tensor constant with the same initialization value copied across
            //         the tensor. This constructor is to support deserialization of constants.
            ///
            /// \param type The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param data A void* to constant data.
            Constant(const nnfusion::element::Type& type,
                     const nnfusion::Shape& shape,
                     const void* data)
                : Op("Constant")
                , m_element_type(type)
                , m_shape(shape)
                , m_data(nullptr)
            {
                size_t size = nnfusion::shape_size(m_shape) * m_element_type.size();
                m_data = nnfusion::aligned_alloc(m_element_type.size(), size);
                std::memcpy(m_data, data, size);
            }

            virtual ~Constant() override;

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override
            {
                infer_element_type();
                gnode->set_output_type_and_shape(0, m_element_type, m_shape);
            }

            /// \brief Wrapper around constructing a shared_ptr of a Constant
            ///
            /// \param type The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param values A vector of values to use as the constant data.
            template <typename T>
            static std::shared_ptr<op::Constant> create(const nnfusion::element::Type& type,
                                                        nnfusion::Shape shape,
                                                        const std::vector<T> values)
            {
                auto result = std::make_shared<op::Constant>(type, shape, values);
                //result->validate_and_infer_types();
                return result;
            }

            /// \brief Wrapper around constructing a shared_ptr of a Constant
            ///
            /// \param type The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param values An initializer_list of values to use as the constant data.
            template <typename T>
            static std::shared_ptr<op::Constant> create(const nnfusion::element::Type& type,
                                                        nnfusion::Shape shape,
                                                        std::initializer_list<T> values)
            {
                auto result = std::make_shared<op::Constant>(type, shape, std::vector<T>{values});
                //result->validate_and_infer_types();
                return result;
            }

            /// \return The initialization literals for the tensor constant.
            std::vector<std::string> get_value_strings() const;

            template <typename T>
            std::vector<T> get_vector() const
            {
                CHECK(sizeof(T) <= m_element_type.size() || nnfusion::shape_size(m_shape) << 0)
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
            Constant(const std::string& name)
                : Op(name)
                , m_shape({})
            {
            }

            virtual void infer_element_type() {}
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
                CHECK(source.size() == target_element_count)
                    << "Constant initializer does not match shape";

                if (target_type == nnfusion::element::boolean)
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
                    CHECK_FAIL() << "unsupported type";
                }
            }

            bool m_is_weight = false;
            nnfusion::element::Type m_element_type;
            nnfusion::Shape m_shape{};
            void* m_data{nullptr};
            Constant(const Constant&) = delete;
            Constant(Constant&&) = delete;
            Constant operator=(const Constant*) = delete;
        };

        class ScalarConstantLikeBase : public Constant
        {
        public:
            std::shared_ptr<Constant> as_constant() const;

        protected:
            ScalarConstantLikeBase(const std::string& name)
                : Constant(name)
            {
            }
        };

        /// \brief A scalar constant whose element type is the same as like.
        template <typename T>
        class ScalarConstantLike : public ScalarConstantLikeBase
        {
        public:
            /// \brief A scalar constant whose element type is the same as like.
            ///
            /// Once the element type is known, the dependency on like will be removed and
            /// this node will be replaced with an equivalent constant.
            ///
            /// \param like A tensor that will supply the element type.
            /// \param value The value of the scalar.
            ScalarConstantLike(const std::shared_ptr<graph::GNode>& like, T value);

        protected:
            void infer_element_type() override
            {
                if (nullptr == m_data)
                {
                    m_data = nnfusion::aligned_alloc(m_element_type.size(), m_element_type.size());
                    write_values(std::vector<T>(1, m_value));
                }
            }

            T m_value;
        };
    }
}
