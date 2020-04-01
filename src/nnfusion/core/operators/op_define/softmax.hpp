// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Softmax operation.
        ///
        class Softmax : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a softmax operation.
            ///
            /// \param axes The axis positions (0-based) on which to calculate the softmax.
            Softmax(const nnfusion::AxisSet& axes);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            const nnfusion::AxisSet& get_axes() const { return m_axes; }
        private:
            nnfusion::AxisSet m_axes;
        };
    }
}
