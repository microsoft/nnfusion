// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "../op.hpp"

namespace nnfusion
{
    namespace op
    {
        class IndexReduction : public Op
        {
        public:
            size_t get_reduction_axis() const { return m_axis; }
            nnfusion::element::Type get_index_element_type() const { return m_index_element_type; }
            IndexReduction(const std::string& node_type,
                           size_t axis,
                           const nnfusion::element::Type& index_element_type);

        protected:
            size_t m_axis;
            nnfusion::element::Type m_index_element_type;

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;
        };
    }
}
