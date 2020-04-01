// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/op.hpp"

namespace nnfusion
{
    namespace op
    {
        //brief Computes indices of top k maximum/minimum index along a specified axis for a given tensor
        class TopK : public Op
        {
        public:
            /// \brief Constructs a TopK operation.
            ///
            /// \param top_k_axis The axis along which to compute top k indices
            /// \param index_element_type produce indices. Currently, only int64 or int32 are supported
            /// \param k Number of top indices to compute. Compute all indices if k = 0
            /// \param compute_max Compute top k max or top k min?
            TopK(size_t top_k_axis,
                 const nnfusion::element::Type& index_element_type,
                 size_t k = 0,
                 bool compute_max = true);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            size_t get_top_k_axis() const { return m_top_k_axis; }
            nnfusion::element::Type get_index_element_type() const { return m_index_element_type; }
            size_t get_k() const { return m_k; }
            bool get_compute_max() const { return m_compute_max; }
        protected:
            size_t m_top_k_axis;
            nnfusion::element::Type m_index_element_type;
            size_t m_k;
            bool m_compute_max;
        };
    }
}
