//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <limits>
#include <stddef.h>
#include "nnfusion/common/shape.hpp"
#include "nnfusion/util/util.hpp"
using namespace nnfusion;

namespace nnfusion
{
    namespace
    {
        std::vector<std::string> split_string(const std::string& s, const std::string& delimiter)
        {
            size_t last = 0;
            size_t next = 0;
            std::vector<std::string> result;
            while ((next = s.find(delimiter, last)) != std::string::npos)
            {
                result.push_back(s.substr(last, next - last));
                last = next + 1;
            }
            result.push_back(s.substr(last, next - last));
            return result;
        }
    } // namespace

    /// \brief Class representing a dimension, which may be symbolic or dynamic (undetermined until runtime),
    ///        in a shape.
    class SymDim
    {
    public:
        /// \brief Construct a static dimension.
        SymDim(size_t dimension)
            : m_sym("")
            , m_min(0)
            , m_max(dimension)
        {
        }

        /// \brief Construct a sybolic dimension.
        SymDim(std::string dimension, size_t min, size_t max)
            : m_sym(dimension)
            , m_min(min)
            , m_max(max)
        {
        }
        SymDim(std::string dimension)
        {
            if (dimension.find(":") == std::string::npos)
            {
                m_sym = dimension;
                m_min = 0;
                m_max = std::numeric_limits<size_t>::max();
            }
            else
            {
                auto dim_info = split_string(dimension, ":");
                NNFUSION_CHECK((dim_info.size() > 1 && dim_info.size() < 4)) << "illegal dim info "
                                                                             << dimension;
                if (dim_info.size() == 2)
                {
                    m_sym = dim_info[0];
                    m_min = 0;
                    m_max = std::stoull(dim_info[1]);
                }
                else if (dim_info.size() == 3)
                {
                    m_sym = dim_info[0];
                    m_min = std::stoull(dim_info[1]);
                    m_max = std::stoull(dim_info[2]);
                }
            }
        }

        std::string get_name() { if(m_name.length()==0) return sym(); return m_name; }
        void set_name(std::string name) { m_name = name; }

        SymDim() {}
        bool is_dynamic() const { return m_sym.size() > 0; }
        bool is_static() const { return !is_dynamic(); }
        std::string sym() const { return m_sym; }
        size_t min() const { return m_min; }
        size_t max() const { return m_max; }
        std::string expr_to_symbol(std::string expr) const
        {
            bool replaced = false;
            int pos;
            while ((pos = expr.find("+")) != std::string::npos)
            {
                expr.replace(pos, 1, "a");
                replaced = true;
            }
            while ((pos = expr.find("-")) != std::string::npos)
            {
                expr.replace(pos, 1, "s");
                replaced = true;
            }
            while ((pos = expr.find("*")) != std::string::npos)
            {
                expr.replace(pos, 1, "m");
                replaced = true;
            }
            return replaced ? "s" + expr : expr;
        }

        std::string to_string() const
        {
            if (is_static())
            {
                return std::to_string(m_max);
            }
            else
            {
                if (m_max < std::numeric_limits<size_t>::max())
                    return "\"" + expr_to_symbol(m_sym) + ":" + std::to_string(m_max) + "\"";
                else
                    return "\"" + expr_to_symbol(m_sym) + "\"";
            }
        }
        std::string debug_string() const
        {
            if (is_static())
            {
                return std::to_string(m_max);
            }
            else
            {
                if (m_min > 0)
                    return m_sym + ":" + std::to_string(m_min) + ":" + std::to_string(m_max);
                else if (m_max < std::numeric_limits<size_t>::max())
                    return m_sym + ":" + std::to_string(m_max);
                else
                    return m_sym;
            }
        }

        /// \brief Addition operator for Dimension.
        SymDim operator+(const SymDim& dim) const
        {
            if (is_static() && dim.is_static())
                return SymDim(m_max + dim.max());
            else if (is_static() && dim.is_dynamic())
                return SymDim(
                    dim.sym() + "+" + std::to_string(m_max), m_max + dim.min(), m_max + dim.max());
            else if (is_dynamic() && dim.is_static())
                return SymDim(
                    m_sym + "+" + std::to_string(dim.max()), m_min + dim.max(), m_max + dim.max());
            else
                return SymDim(m_sym + "+" + dim.sym(), m_min + dim.min(), m_max + dim.max());
        }

        size_t safesub(const size_t a, const size_t b) const { return (a > b) ? a - b : 0; }
        /// \brief Subtraction operator for Dimension.
        SymDim operator-(const SymDim& dim) const
        {
            if (is_static() && dim.is_static())
                return SymDim(m_max - dim.max());
            else if (is_static() && dim.is_dynamic())
                return SymDim(std::to_string(m_max) + "-" + dim.sym(),
                              safesub(m_max, dim.max()),
                              safesub(m_max, dim.min()));
            else if (is_dynamic() && dim.is_static())
                return SymDim(m_sym + "-" + std::to_string(dim.max()),
                              safesub(m_min, dim.max()),
                              safesub(m_max, dim.max()));
            else
                return SymDim(
                    m_sym + "-" + dim.sym(), safesub(m_min, dim.max()), safesub(m_max, dim.min()));
        }

        /// \brief Multiplication operator for Dimension.
        SymDim operator*(const SymDim& dim) const
        {
            if (is_static() && dim.is_static())
                return SymDim(m_max * dim.max());
            else if (is_static() && dim.is_dynamic())
                return SymDim(
                    std::to_string(m_max) + "*" + dim.sym(), m_max * dim.min(), m_max * dim.max());
            else if (is_dynamic() && dim.is_static())
                return SymDim(
                    m_sym + "*" + std::to_string(dim.max()), m_min * dim.max(), m_max * dim.max());
            else
                return SymDim(m_sym + "*" + dim.sym(), m_min * dim.min(), m_max * dim.max());
        }

        /// \brief Add-into operator for Dimension.
        SymDim& operator+=(const SymDim& dim) { return (*this = *this + dim); }
        /// \brief Multiply-into operator for Dimension.
        SymDim& operator*=(const SymDim& dim) { return (*this = *this * dim); }
        bool operator<(const SymDim &other) const { return this->m_sym < other.sym(); }  
    private:
        // the symbol name of this dim
        std::string m_sym;
        // if m_sym is not empty, [m_min, m_max] are optional value to represnet the range of m_sym;
        // otherwise, the m_max is use to represent the numeric value of this dim.
        size_t m_min;
        size_t m_max;
        std::string m_name;
    };

    /// \brief Insert a human-readable representation of a dimension into an output stream.
    std::ostream& operator<<(std::ostream& str, const SymDim& dimension);

    /// \brief Symbolic Shape for a tensor.
    class SymShape : public std::vector<SymDim>
    {
    public:
        SymShape(const std::initializer_list<SymDim>& dims)
            : std::vector<SymDim>(dims)
        {
        }

        SymShape(const std::vector<SymDim>& dims)
            : std::vector<SymDim>(dims)
        {
        }

        SymShape(const Shape& axis_lengths)
        {
            for (auto d : axis_lengths)
            {
                this->push_back(SymDim(d));
            }
        }

        SymShape(const SymShape& shape) { (*this) = shape; }
        Shape to_static() const
        {
            Shape ret;
            for (auto d : *this)
            {
                ret.push_back(d.max());
            }
            return ret;
        }

        explicit SymShape(size_t n, size_t initial_value = 0)
            : std::vector<SymDim>(n, SymDim(initial_value))
        {
        }

        template <class InputIterator>
        SymShape(InputIterator first, InputIterator last)
            : std::vector<SymDim>(first, last)
        {
        }

        SymShape() {}
        SymShape& operator=(const SymShape& v)
        {
            static_cast<std::vector<SymDim>*>(this)->operator=(v);
            return *this;
        }
        SymShape& operator=(SymShape&& v)
        {
            static_cast<std::vector<SymDim>*>(this)->operator=(v);
            return *this;
        }

        bool is_static() const
        {
            for (auto dim : *this)
                if (dim.is_dynamic())
                    return false;
            return true;
        }

        bool is_dynamic() const { return !is_static(); }
    };

    std::ostream& operator<<(std::ostream& s, const SymShape& shape);
}
