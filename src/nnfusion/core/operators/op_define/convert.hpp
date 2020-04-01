// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "../op.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise type conversion operation.
        class Convert : public Op
        {
        public:
            /// \brief Constructs a conversion operation.
            ///
            /// \param element_type Element type for the output tensor.
            Convert(const nnfusion::element::Type& element_type);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            const nnfusion::element::Type& get_convert_element_type() const
            {
                return m_element_type;
            }

        protected:
            const nnfusion::element::Type m_element_type;
        };
    }
}
