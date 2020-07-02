// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/common/descriptor/layout/tensor_layout.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/common/type/element_type.hpp"

nnfusion::descriptor::layout::TensorLayout::TensorLayout(const descriptor::Tensor& tensor)
    : m_element_type(tensor.get_element_type())
    , m_shape(tensor.get_shape())
{
}

const nnfusion::element::Type& nnfusion::descriptor::layout::TensorLayout::get_element_type() const
{
    return m_element_type;
}

const nnfusion::Shape& nnfusion::descriptor::layout::TensorLayout::get_shape() const
{
    return m_shape;
}

size_t nnfusion::descriptor::layout::TensorLayout::get_size() const
{
    return nnfusion::shape_size(get_shape());
}

size_t nnfusion::descriptor::layout::TensorLayout::get_allocated_size()
{
    return get_size() * get_element_type().size();
}
