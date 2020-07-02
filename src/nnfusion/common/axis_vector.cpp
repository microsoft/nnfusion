// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/common/axis_vector.hpp"
#include "nnfusion/common/util.hpp"

std::ostream& nnfusion::operator<<(std::ostream& s, const AxisVector& axis_vector)
{
    s << "AxisVector{";
    s << nnfusion::join(axis_vector);
    s << "}";
    return s;
}
