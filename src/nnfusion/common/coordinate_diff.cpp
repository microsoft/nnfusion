// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/common/coordinate_diff.hpp"
#include "nnfusion/common/util.hpp"

std::ostream& nnfusion::operator<<(std::ostream& s, const CoordinateDiff& coordinate_diff)
{
    s << "CoordinateDiff{";
    s << nnfusion::join(coordinate_diff);
    s << "}";
    return s;
}
