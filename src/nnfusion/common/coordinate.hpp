// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <vector>

#include "nnfusion/common/axis_set.hpp"
#include "nnfusion/common/shape.hpp"

namespace nnfusion
{
    /// \brief Coordinates for a tensor element
    class Coordinate : public std::vector<size_t>
    {
    public:
        Coordinate() {}
        Coordinate(const std::initializer_list<size_t>& axes)
            : std::vector<size_t>(axes)
        {
        }

        Coordinate(const Shape& shape)
            : std::vector<size_t>(static_cast<const std::vector<size_t>&>(shape))
        {
        }

        Coordinate(const std::vector<size_t>& axes)
            : std::vector<size_t>(axes)
        {
        }

        Coordinate(const Coordinate& axes)
            : std::vector<size_t>(axes)
        {
        }

        Coordinate(size_t n, size_t initial_value = 0)
            : std::vector<size_t>(n, initial_value)
        {
        }

        template <class InputIterator>
        Coordinate(InputIterator first, InputIterator last)
            : std::vector<size_t>(first, last)
        {
        }

        Coordinate& operator=(const Coordinate& v)
        {
            static_cast<std::vector<size_t>*>(this)->operator=(v);
            return *this;
        }

        Coordinate& operator=(Coordinate&& v)
        {
            static_cast<std::vector<size_t>*>(this)->operator=(v);
            return *this;
        }
    };

    std::ostream& operator<<(std::ostream& s, const Coordinate& coordinate);
}
