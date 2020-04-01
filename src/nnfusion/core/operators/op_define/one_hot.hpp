// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/op.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief One-hot operator.
        ///
        /// ## Parameters
        ///
        /// |                | Description                                                |
        /// | -------------- | ---------------------------------------------------------- |
        /// | `shape`        | The desired output shape, including the new one-hot axis.  |
        /// | `one_hot_axis` | The index within the output shape of the new one-hot axis. |
        ///
        /// ## Inputs
        ///
        /// |       | Type                                                    | Description                                 |
        /// | ----- | ------------------------------------------------------- | ------------------------------------------- |
        /// | `arg` | \f$E[d_1,\dots,d_{m-1},d_{m+1},\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and any element type. |
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                                                                                                                                                                                                                                |
        /// | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        /// | \f$E[d_1,\dots,d_n]\f$ | The tensor \f$T'\f$, where \f$T'[i_1,\dots,i_{m-1},i_m,i_{m+1},\dots,i_n] = 1\f$ if \f$T[i_1,\dots,i_{m-1},i_{m+1},\dots,i_n] = i_m\f$, else \f$0\f$. However, \f$T'\f$ is undefined if any non-integral value or any out-of-bounds value is detected in the input tensor. |
        class OneHot : public Op
        {
        public:
            /// \brief Constructs a one-hot operation.
            ///
            /// \param shape        The shape of the output tensor, including the new one-hot axis.
            /// \param one_hot_axis The index within the output shape of the new one-hot axis.
            OneHot(const nnfusion::PartialShape& shape, size_t one_hot_axis);

            /// \return The index of the one-hot axis.
            size_t get_one_hot_axis() const { return m_one_hot_axis; }
        protected:
            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            nnfusion::PartialShape m_shape;
            size_t m_one_hot_axis;
        };
    }
}
