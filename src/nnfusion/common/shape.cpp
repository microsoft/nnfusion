// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/common/shape.hpp"
#include "nnfusion/common/util.hpp"

std::ostream& nnfusion::operator<<(std::ostream& s, const Shape& shape)
{
    s << "Shape{";
    s << nnfusion::join(shape);
    s << "}";
    return s;
}
