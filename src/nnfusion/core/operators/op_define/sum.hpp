// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/arithmetic_reduction.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Tensor sum operation.
        ///
        /// Element-wise sums the input tensor, eliminating the specified reduction axes.
        /// For example:
        ///
        /// \f[
        ///     \mathit{sum}\left(\{0\},
        ///         \left[ \begin{array}{ccc}
        ///                1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
        ///     \left[ (1 + 3 + 5), (2 + 4 + 6) \right] =
        ///     \left[ 9, 12 \right]~~~\text{(dimension 0 (rows) is eliminated)}
        /// \f]
        ///
        /// \f[
        ///     \mathit{sum}\left(\{1\},
        ///         \left[ \begin{array}{ccc}
        ///                1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
        ///     \left[ (1 + 2), (3 + 4), (5 + 6) \right] =
        ///     \left[ 3, 7, 11 \right]~~~\text{(dimension 1 (columns) is eliminated)}
        /// \f]
        ///
        /// \f[
        ///     \mathit{sum}\left(\{0,1\},
        ///         \left[ \begin{array}{ccc}
        ///                1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
        ///      (1 + 2) + (3 + 4) + (5 + 6) =
        ///      21~~~\text{(both dimensions (rows and columns) are eliminated)}
        /// \f]
        ///
        /// This is equivalent to Reduce where `arg_init` = 0 and `reduction_graph` is \f$f(x,y) = x+y\f$.
        ///
        /// ## Parameters
        ///
        /// |                      | Description                              |
        /// | -------------------- | ---------------------------------------- |
        /// | `reduction_axes`     | The axes to eliminate through summation. |
        ///
        /// ## Output
        ///
        /// | Type                                      | Description                                                                                                      |
        /// | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
        /// | \f$N[\textit{delete}(A,d_1,\dots,d_n)]\f$ | The tensor \f$T\f$, where \f$T\f$ is the input tensor with the `reduction_axes` \f$A\f$ eliminated by summation. |
        class Sum : public ArithmeticReduction
        {
        public:
            /// \brief Constructs a summation operation.
            ///
            /// \param reduction_axes The axis positions (0-based) to be eliminated.
            Sum(const nnfusion::AxisSet& reduction_axes);
        };
    }
}
