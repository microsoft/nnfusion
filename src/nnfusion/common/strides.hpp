// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <ostream>
#include <vector>

namespace nnfusion
{
    /// \brief Strides for a tensor.
    class Strides : public std::vector<size_t>
    {
    public:
        Strides(const std::initializer_list<size_t>& axis_strides)
            : std::vector<size_t>(axis_strides)
        {
        }

        Strides(const std::vector<size_t>& axis_strides)
            : std::vector<size_t>(axis_strides)
        {
        }

        Strides(const Strides& axis_strides)
            : std::vector<size_t>(axis_strides)
        {
        }

        explicit Strides(size_t n, size_t initial_value = 0)
            : std::vector<size_t>(n, initial_value)
        {
        }

        template <class InputIterator>
        Strides(InputIterator first, InputIterator last)
            : std::vector<size_t>(first, last)
        {
        }

        Strides() {}
        Strides& operator=(const Strides& v)
        {
            static_cast<std::vector<size_t>*>(this)->operator=(v);
            return *this;
        }
        Strides& operator=(Strides&& v)
        {
            static_cast<std::vector<size_t>*>(this)->operator=(v);
            return *this;
        }
    };

    std::ostream& operator<<(std::ostream& s, const Strides& strides);
}
