// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/common/axis_set.hpp"
#include "nnfusion/common/util.hpp"

std::ostream& nnfusion::operator<<(std::ostream& s, const AxisSet& axis_set)
{
    s << "AxisSet{";
    s << nnfusion::join(axis_set);
    s << "}";
    return s;
}
