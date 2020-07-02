// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <ostream>
#include <vector>

namespace nnfusion
{
    /// \brief A vector of axes.
    class AxisVector : public std::vector<size_t>
    {
    public:
        AxisVector(const std::initializer_list<size_t>& axes)
            : std::vector<size_t>(axes)
        {
        }

        AxisVector(const std::vector<size_t>& axes)
            : std::vector<size_t>(axes)
        {
        }

        AxisVector(const AxisVector& axes)
            : std::vector<size_t>(axes)
        {
        }

        explicit AxisVector(size_t n)
            : std::vector<size_t>(n)
        {
        }

        template <class InputIterator>
        AxisVector(InputIterator first, InputIterator last)
            : std::vector<size_t>(first, last)
        {
        }

        AxisVector() {}
        AxisVector& operator=(const AxisVector& v)
        {
            static_cast<std::vector<size_t>*>(this)->operator=(v);
            return *this;
        }
        AxisVector& operator=(AxisVector&& v)
        {
            static_cast<std::vector<size_t>*>(this)->operator=(v);
            return *this;
        }
    };

    std::ostream& operator<<(std::ostream& s, const AxisVector& axis_vector);
}
