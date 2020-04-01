// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "../op.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief A graph parameter.
        ///
        /// Parameters are nodes that represent the arguments that will be passed to user-defined graphs.
        /// Function creation requires a sequence of parameters.
        /// Basic graph operations do not need parameters attached to a graph.
        class Parameter : public Op
        {
        public:
            /// \brief Constructions a tensor view-typed parameter node.
            ///
            /// \param element_type The element type of the parameter.
            /// \param pshape The partial shape of the parameter.
            /// \param cacheable True if the parameter is not expected to be frequently updated.
            Parameter(const nnfusion::element::Type& element_type,
                      const nnfusion::PartialShape& pshape,
                      const bool cacheable = false);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            bool get_cacheable() const { return m_cacheable; }
            bool is_parameter() const override { return true; }
        protected:
            bool m_cacheable;
            nnfusion::PartialShape m_partial_shape;
            nnfusion::element::Type m_element_type;
        };
    }
}
