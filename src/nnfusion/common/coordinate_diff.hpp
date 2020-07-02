// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <ostream>
#include <vector>

namespace nnfusion
{
    /// \brief A difference (signed) of tensor element coordinates.
    class CoordinateDiff : public std::vector<std::ptrdiff_t>
    {
    public:
        CoordinateDiff(const std::initializer_list<std::ptrdiff_t>& diffs)
            : std::vector<std::ptrdiff_t>(diffs)
        {
        }

        CoordinateDiff(const std::vector<std::ptrdiff_t>& diffs)
            : std::vector<std::ptrdiff_t>(diffs)
        {
        }

        CoordinateDiff(const CoordinateDiff& diffs)
            : std::vector<std::ptrdiff_t>(diffs)
        {
        }

        explicit CoordinateDiff(size_t n, std::ptrdiff_t initial_value = 0)
            : std::vector<std::ptrdiff_t>(n, initial_value)
        {
        }

        template <class InputIterator>
        CoordinateDiff(InputIterator first, InputIterator last)
            : std::vector<std::ptrdiff_t>(first, last)
        {
        }

        CoordinateDiff() {}
        CoordinateDiff& operator=(const CoordinateDiff& v)
        {
            static_cast<std::vector<std::ptrdiff_t>*>(this)->operator=(v);
            return *this;
        }
        CoordinateDiff& operator=(CoordinateDiff&& v)
        {
            static_cast<std::vector<std::ptrdiff_t>*>(this)->operator=(v);
            return *this;
        }
    };

    std::ostream& operator<<(std::ostream& s, const CoordinateDiff& coordinate_diff);
}
