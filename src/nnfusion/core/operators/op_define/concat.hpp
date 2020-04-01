// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/op.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Concatenation operation.
        class Concat : public Op
        {
        public:
            /// \brief Constructs a concatenation operation.
            ///
            /// \param concatenation_axis The axis along which to concatenate the input tensors.
            Concat(size_t concatenation_axis);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            /// \return The concatenation axis.
            size_t get_concatenation_axis() const { return m_concatenation_axis; }
        protected:
            const size_t m_concatenation_axis;
        };
    }
}
