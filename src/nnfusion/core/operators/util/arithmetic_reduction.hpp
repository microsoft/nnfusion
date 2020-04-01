// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "../op.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Abstract base class for arithmetic reduction operations, i.e., operations where chosen axes of the input tensors
        ///        are eliminated (reduced out) by repeated application of a particular binary arithmetic operation.
        class ArithmeticReduction : public Op
        {
        public:
            /// \brief Constructs an arithmetic reduction operation.
            ///
            /// \param reduction_axes The axis positions (0-based) to be eliminated.
            ArithmeticReduction(const std::string& node_type,
                                const nnfusion::AxisSet& reduction_axes);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            /// \return The axis positions (0-based) to be eliminated through reduction.
            const nnfusion::AxisSet& get_reduction_axes() const { return m_reduction_axes; }
        protected:
            nnfusion::AxisSet m_reduction_axes;
        };
    }
}
