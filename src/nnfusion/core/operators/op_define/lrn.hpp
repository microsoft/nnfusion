// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise Local Response Normalization (LRN) operation.
        class LRN : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a LRN operation.
            LRN(double alpha, double beta, double bias, size_t size);

            double get_alpha() const { return m_alpha; }
            double get_beta() const { return m_beta; }
            double get_bias() const { return m_bias; }
            size_t get_nsize() const { return m_size; }
            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

        protected:
            double m_alpha;
            double m_beta;
            double m_bias;
            size_t m_size;
        };
    }
}
