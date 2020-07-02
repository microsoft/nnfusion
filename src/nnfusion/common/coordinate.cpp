// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/common/coordinate.hpp"
#include "nnfusion/common/util.hpp"

std::ostream& nnfusion::operator<<(std::ostream& s, const Coordinate& coordinate)
{
    s << "Coordinate{";
    s << nnfusion::join(coordinate);
    s << "}";
    return s;
}
