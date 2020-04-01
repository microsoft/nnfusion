// Microsoft (c) 2019, NNFUSION Team
#include "reference_common.hpp"

namespace nnfusion
{
    namespace kernels
    {
        LanguageUnit_p reference_common_header =
            LanguageUnit_p(new LanguageUnit("reference_common.h",
                                            R"(
#pragma once

#include <algorithm>
#include <set>
#include <vector>

namespace reference_common
{
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

    class Shape : public std::vector<size_t>
    {
    public:
        Shape(const std::initializer_list<size_t>& axis_lengths)
            : std::vector<size_t>(axis_lengths)
        {
        }

        Shape(const std::vector<size_t>& axis_lengths)
            : std::vector<size_t>(axis_lengths)
        {
        }

        Shape(const Shape& axis_lengths)
            : std::vector<size_t>(axis_lengths)
        {
        }

        explicit Shape(size_t n, size_t initial_value = 0)
            : std::vector<size_t>(n, initial_value)
        {
        }

        template <class InputIterator>
        Shape(InputIterator first, InputIterator last)
            : std::vector<size_t>(first, last)
        {
        }

        Shape() {}
        Shape& operator=(const Shape& v)
        {
            static_cast<std::vector<size_t>*>(this)->operator=(v);
            return *this;
        }
        Shape& operator=(Shape&& v)
        {
            static_cast<std::vector<size_t>*>(this)->operator=(v);
            return *this;
        }
    };

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

    template <typename T>
    T ceil_div(const T& x, const T& y)
    {
        return (x == 0 ? 0 : (1 + (x - 1) / y));
    }

    class CoordinateTransform
    {
    public:
        CoordinateTransform(const Shape& source_shape,
                            const Coordinate& source_start_corner,
                            const Coordinate& source_end_corner,
                            const Strides& source_strides,
                            const AxisVector& source_axis_order,
                            const CoordinateDiff& target_padding_below,
                            const CoordinateDiff& target_padding_above,
                            const Strides& target_dilation_strides)
            : m_source_shape(source_shape)
                , m_source_start_corner(source_start_corner)
                , m_source_end_corner(source_end_corner)
                , m_source_strides(source_strides)
                , m_source_axis_order(source_axis_order)
                , m_target_padding_below(target_padding_below)
                , m_target_padding_above(target_padding_above)
                , m_target_dilation_strides(target_dilation_strides)
        {
            m_n_axes = source_shape.size();
            for (size_t axis = 0; axis < m_n_axes; axis++)
            {
                m_target_shape.push_back(ceil_div(source_end_corner[source_axis_order[axis]] -
                                                      source_start_corner[source_axis_order[axis]],
                                                  source_strides[source_axis_order[axis]]));
            }
        }

        CoordinateTransform(const Shape& source_shape,
                            const Coordinate& source_start_corner,
                            const Coordinate& source_end_corner,
                            const Strides& source_strides,
                            const AxisVector& source_axis_order,
                            const CoordinateDiff& target_padding_below,
                            const CoordinateDiff& target_padding_above)
            : CoordinateTransform(source_shape,
                                  source_start_corner,
                                  source_end_corner,
                                  source_strides,
                                  source_axis_order,
                                  target_padding_below,
                                  target_padding_above,
                                  default_strides(source_shape.size())){};

        CoordinateTransform(const Shape& source_shape,
                            const Coordinate& source_start_corner,
                            const Coordinate& source_end_corner,
                            const Strides& source_strides,
                            const AxisVector& source_axis_order)
            : CoordinateTransform(source_shape,
                                  source_start_corner,
                                  source_end_corner,
                                  source_strides,
                                  source_axis_order,
                                  default_padding(source_shape.size()),
                                  default_padding(source_shape.size()),
                                  default_strides(source_shape.size())){};

        CoordinateTransform(const Shape& source_shape,
                            const Coordinate& source_start_corner,
                            const Coordinate& source_end_corner,
                            const Strides& source_strides)
            : CoordinateTransform(source_shape,
                                  source_start_corner,
                                  source_end_corner,
                                  source_strides,
                                  default_axis_order(source_shape.size()),
                                  default_padding(source_shape.size()),
                                  default_padding(source_shape.size()),
                                  default_strides(source_shape.size())){};

        CoordinateTransform(const Shape& source_shape,
                            const Coordinate& source_start_corner,
                            const Coordinate& source_end_corner)
            : CoordinateTransform(source_shape,
                                  source_start_corner,
                                  source_end_corner,
                                  default_strides(source_shape.size()),
                                  default_axis_order(source_shape.size()),
                                  default_padding(source_shape.size()),
                                  default_padding(source_shape.size()),
                                  default_strides(source_shape.size())){};

        CoordinateTransform(const Shape& source_shape): CoordinateTransform(source_shape,
                          default_source_start_corner(source_shape.size()),
                          default_source_end_corner(source_shape),
                          default_strides(source_shape.size()),
                          default_axis_order(source_shape.size()),
                          default_padding(source_shape.size()),
                          default_padding(source_shape.size()),
                          default_strides(source_shape.size()))
        {
        };

        size_t index(const Coordinate& c) const { return index_source(to_source_coordinate(c)); };
        bool has_source_coordinate(const Coordinate& c_target) const
        {
            for (size_t target_axis = 0; target_axis < m_n_axes; target_axis++)
            {
                // Is this coordinate out of bounds of the target space?
                if (c_target[target_axis] >= m_target_shape[target_axis])
                {
                    return false;
                }

                // The rest of this is a replay of the corresponding logic in `to_source_coordinate`, with
                // bounds and divisibility checking.
                std::ptrdiff_t source_axis = m_source_axis_order[target_axis];

                std::ptrdiff_t target_pos = c_target[target_axis];
                std::ptrdiff_t pos_destrided = target_pos * m_source_strides[source_axis];
                std::ptrdiff_t pos_deshifted = pos_destrided + m_source_start_corner[source_axis];

                // If we are in the below-padding or the above-padding.
                if (pos_deshifted < m_target_padding_below[target_axis])
                {
                    return false;
                }
                std::ptrdiff_t pos_depadded = pos_deshifted - m_target_padding_below[target_axis];

                // If we are in the above-padding, we have no source coordinate.
                if (m_source_shape[source_axis] == 0 ||
                    (pos_depadded >=
                     ((m_source_shape[source_axis] - 1) * m_target_dilation_strides[target_axis]) +
                         1))
                {
                    return false;
                }

                // If we are in a dilation gap, we have no source coordinate.
                if (pos_depadded % m_target_dilation_strides[target_axis] != 0)
                {
                    return false;
                }
            }

            return true;
        };
        Coordinate to_source_coordinate(const Coordinate& c_target) const
        {
            Coordinate c_source(c_target.size());

            for (size_t target_axis = 0; target_axis < m_n_axes; target_axis++)
            {
                size_t source_axis = m_source_axis_order[target_axis];

                size_t target_pos = c_target[target_axis];
                size_t pos_destrided = target_pos * m_source_strides[source_axis];
                size_t pos_deshifted = pos_destrided + m_source_start_corner[source_axis];
                size_t pos_depadded = pos_deshifted - m_target_padding_below[target_axis];
                size_t pos_dedilated = pos_depadded / m_target_dilation_strides[target_axis];
                c_source[source_axis] = pos_dedilated;
            }

            return c_source;
        };
        const Shape& get_target_shape() const { return m_target_shape; };
        const Shape& get_source_shape() const { return m_source_shape; }
        const Coordinate& get_source_start_corner() const { return m_source_start_corner; }
        const Coordinate& get_source_end_corner() const { return m_source_end_corner; }
        const Strides& get_source_strides() const { return m_source_strides; }
        const AxisVector& get_source_axis_order() const { return m_source_axis_order; }
        const Strides& get_target_dilation_strides() const { return m_target_dilation_strides; }
        class Iterator
        {
        public:
            Iterator(const Shape& target_shape, bool is_end = false)
                : m_target_shape(target_shape)
            {
                // Initial coordinate is (0,...,0) in the target space.
                m_coordinate = Coordinate(target_shape.size(), 0);

                // The case where we have a zero-length axis is a bit special, in that
                // the iterator always starts out of bounds.
                m_empty = false;

                for (auto s : target_shape)
                {
                    if (s == 0)
                    {
                        m_empty = true;
                        break;
                    }
                }

                m_oob = is_end || m_empty;
            };

            void operator++()
            {
                // If we are out of bounds, start over at (0,...0). (TODO: not sure if that's what we want. It might be best to stay put?)
                if (m_oob)
                {
                    std::fill(m_coordinate.begin(), m_coordinate.end(), 0);
                    m_oob = m_empty;
                    return;
                }

                // Increment the target coordinate.
                for (size_t axis = m_target_shape.size(); axis-- > 0;)
                {
                    m_coordinate[axis]++;

                    if (m_coordinate[axis] < m_target_shape[axis])
                    {
                        // No carry-out, so we are done.
                        return;
                    }
                    else
                    {
                        m_coordinate[axis] = 0;
                    }
                }

                // If we are still here there was carry-out from the most significant axis. We are now out of bounds.
                m_oob = true;
            };
            Iterator operator++(int)
            {
                CoordinateTransform::Iterator temp = *this;
                ++(*this);
                return temp;
            };
            void operator+=(size_t n)
            {
                for (size_t i = 0; i < n; i++)
                {
                    ++(*this);
                }
            };
            const Coordinate& operator*() const { return m_coordinate; };
            bool operator!=(const Iterator& it) { return !(*this == it); };
            bool operator==(const Iterator& it)
            {
                if (it.m_oob)
                {
                    // Out-of-bounds iterators are always equal; in other words, an iterator is always equal to
                    // end() even if the internally stored coordinates are different.

                    // If one iterator is out of bounds and the other is not, they are unequal even if their
                    // target coordinates happen to match.
                    return m_oob;
                }
                else if (m_oob)
                {
                    return false;
                }

                if (m_target_shape != it.m_target_shape)
                {
                    return false;
                }

                // Check axis-wise if the iterators are on the same target coordinate.
                for (size_t axis = 0; axis < m_target_shape.size(); axis++)
                {
                    if (m_coordinate[axis] != it.m_coordinate[axis])
                    {
                        return false;
                    }
                }

                return true;
            };

        private:
            Shape m_target_shape;
            Shape m_axis_walk_order;
            Coordinate m_coordinate;
            bool m_oob;
            bool m_empty;
        };

        Iterator begin() noexcept { return Iterator(m_target_shape); }
        Iterator end() noexcept { return Iterator(m_target_shape, true); }
    private:
        size_t index_source(const Coordinate& c) const
        {
            size_t index = 0;
            size_t stride = 1;

            for (size_t axis = m_n_axes; axis-- > 0;)
            {
                index += c[axis] * stride;
                stride *= m_source_shape[axis];
            }

            return index;
        };
        static Strides default_strides(size_t n_axes) { return Strides(n_axes, 1); };
        static CoordinateDiff default_padding(size_t n_axes) { return CoordinateDiff(n_axes, 0); };
        static AxisVector default_axis_order(size_t n_axes)
        {
            AxisVector result(n_axes);
            size_t n = 0;
            std::generate(result.begin(), result.end(), [&n]() -> size_t { return n++; });

            return result;
        };
        static Coordinate default_source_start_corner(size_t n_axes)
        {
            return Coordinate(n_axes, 0);
        };
        static Coordinate default_source_end_corner(const Shape& source_shape)
        {
            return source_shape;
        };

        Shape m_source_shape;
        Coordinate m_source_start_corner;
        Coordinate m_source_end_corner;
        Strides m_source_strides;
        AxisVector m_source_axis_order;
        CoordinateDiff m_target_padding_below;
        CoordinateDiff m_target_padding_above;
        Strides m_target_dilation_strides;

        Shape m_target_shape;
        size_t m_n_axes;
    };

    template <typename AXIS_VALUES>
    AXIS_VALUES project(const AXIS_VALUES& axis_values, const AxisSet& axes)
    {
        AXIS_VALUES result;

        for (size_t i = 0; i < axis_values.size(); i++)
        {
            if (axes.find(i) != axes.end())
            {
                result.push_back(axis_values[i]);
            }
        }

        return result;
    }

    template <typename SHAPE_TYPE>
    size_t shape_size(const SHAPE_TYPE& shape)
    {
        size_t size = 1;
        for (auto d : shape)
        {
            size *= d;
        }
        return size;
    }

    // Removes some values from a vector of axis values
    template <typename AXIS_VALUES>
    AXIS_VALUES reduce(const AXIS_VALUES& axis_values, const AxisSet& deleted_axes)
    {
        AxisSet axes;

        for (size_t i = 0; i < axis_values.size(); i++)
        {
            if (deleted_axes.find(i) == deleted_axes.end())
            {
                axes.insert(i);
            }
        }

        return project(axis_values, axes);
    }
}
)"));
    }
}