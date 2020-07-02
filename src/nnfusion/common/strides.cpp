// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/common/strides.hpp"
#include "nnfusion/common/util.hpp"

std::ostream& nnfusion::operator<<(std::ostream& s, const Strides& strides)
{
    s << "Strides{";
    s << nnfusion::join(strides);
    s << "}";
    return s;
}
