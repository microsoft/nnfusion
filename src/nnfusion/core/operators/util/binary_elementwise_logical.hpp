// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "../op.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Abstract base class for elementwise binary logical operations, i.e., operations where the same
        ///        scalar binary logical operation is applied to each corresponding pair of elements in two same-shaped
        ///        boolean input tensors.
        ///
        /// For example, if the underlying operation (determined by the subclass) is \f$\mathit{op}(x,y)\f$, the input tensors
        /// \f$[[x_0,y_0],[z_0,w_0]]\f$ and \f$[[x_1,y_1],[z_1,w_1]]\f$ will be mapped to \f$[[\mathit{op}(x_0,x_1),\mathit{op}(y_0,y_1)],[\mathit{op}(z_0,z_1),\mathit{op}(w_0,w_1)]]\f$.
        ///
        /// ## Inputs
        ///
        /// |        | Type                                          | Description                                            |
        /// | ------ | --------------------------------------------- | ------------------------------------------------------ |
        /// | `arg0` | \f$\texttt{bool}[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape, with element type `bool`.       |
        /// | `arg1` | \f$\texttt{bool}[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of the same shape and element type as `arg0`. |
        ///
        /// ## Output
        ///
        /// | Type                               | Description                                                                                                                                                                                                        |
        /// | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
        /// | \f$\texttt{bool}[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \mathit{op}(\texttt{arg0}[i_1,\dots,i_n],\texttt{arg1}[i_1,\dots,i_n])\f$. This will always have the same shape as the input tensors, and the element type `bool`. |
        class BinaryElementwiseLogical : public Op
        {
        public:
            /// \brief Constructs a binary elementwise logical operation.
            ///
            BinaryElementwiseLogical(const std::string& node_type);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;
        };
    }
}
