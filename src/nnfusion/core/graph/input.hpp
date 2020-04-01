// Microsoft (c) 2019, NNFusion Team

#pragma once
#include "nnfusion/common/partial_shape.hpp"
#include "nnfusion/common/shape.hpp"
#include "nnfusion/common/type/element_type.hpp"
#include "nnfusion/util/util.hpp"

namespace nnfusion
{
    namespace graph
    {
        class Input
        {
        public:
            Input(const nnfusion::element::Type& element_type, const nnfusion::PartialShape& pshape)
                : m_element_type(element_type)
                , m_shape(pshape.is_static() ? pshape.to_shape() : nnfusion::Shape{})
                , m_partial_shape(pshape)
            {
            }

            const nnfusion::element::Type& get_element_type() const { return m_element_type; }
            const nnfusion::Shape& get_shape() const
            {
                CHECK(m_partial_shape.is_static())
                    << "get_shape was called on a descriptor::Tensor with dynamic shape";
                return m_shape;
            };
            const nnfusion::PartialShape& get_partial_shape() const { return m_partial_shape; }
        private:
            nnfusion::element::Type m_element_type;

            // TODO(amprocte): For now we are maintaining both m_shape and m_partial_shape fields,
            //    with m_shape possibly being invalid (get_shape will throw an exception if it
            //    is). This is because get_shape() returns a const reference. I think ideally we
            //    should refactor so that get_shape returns by value.
            nnfusion::Shape m_shape;
            nnfusion::PartialShape m_partial_shape;

            Input(const Input&) = delete;
            Input(Input&&) = delete;
            Input& operator=(const Input&) = delete;
        };
    }
}