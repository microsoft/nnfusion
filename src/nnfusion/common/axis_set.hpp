// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <ostream>
#include <set>
#include <vector>

namespace nnfusion
{
    /// \brief A set of axes.
    class AxisSet : public std::set<size_t>
    {
    public:
        AxisSet() {}
        AxisSet(const std::initializer_list<size_t>& axes)
            : std::set<size_t>(axes)
        {
        }

        AxisSet(const std::set<size_t>& axes)
            : std::set<size_t>(axes)
        {
        }

        AxisSet(const std::vector<size_t>& axes)
            : std::set<size_t>(axes.begin(), axes.end())
        {
        }

        AxisSet(const AxisSet& axes)
            : std::set<size_t>(axes)
        {
        }

        AxisSet& operator=(const AxisSet& v)
        {
            static_cast<std::set<size_t>*>(this)->operator=(v);
            return *this;
        }

        AxisSet& operator=(AxisSet&& v)
        {
            static_cast<std::set<size_t>*>(this)->operator=(v);
            return *this;
        }
    };

    std::ostream& operator<<(std::ostream& s, const AxisSet& axis_set);
}
