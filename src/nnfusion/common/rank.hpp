// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/common/dimension.hpp"

namespace nnfusion
{
    /// \brief Alias for Dimension, used when the value represents the number of axes in a shape,
    ///        rather than the size of one dimension in a shape.
    ///
    /// XXX: THIS TYPE IS EXPERIMENTAL AND THE ENTIRE DESIGN IS SUBJECT TO CHANGE.
    using Rank = Dimension;
}
