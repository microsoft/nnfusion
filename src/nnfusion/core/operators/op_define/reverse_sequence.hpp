// Microsoft (c) 2019, NNFusion Team

#pragma once

#include <memory>

#include "nnfusion/core/operators/op.hpp"

namespace nnfusion
{
    namespace op
    {
        class ReverseSequence : public Op
        {
        public:
            /// \brief Constructs an arcsin operation.
            ///
            ReverseSequence(size_t batch_axis, size_t seq_axis);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            size_t get_batch_axis() const { return m_batch_axis; }
            size_t get_sequence_axis() const { return m_seq_axis; }
        private:
            size_t m_batch_axis{0};
            size_t m_seq_axis{0};
        };
    }
}
