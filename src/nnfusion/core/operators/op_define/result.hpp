// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "../op.hpp"

namespace nnfusion
{
    namespace op
    {
        class Result : public Op
        {
        public:
            /// \brief Allows a value to be used as a graph result.
            Result();

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            virtual bool is_output() const override { return m_needs_copy_to_host; }
            void set_needs_default_layout(bool val) { m_needs_default_layout = val; }
            bool needs_default_layout() const { return m_needs_default_layout; }
            void set_needs_copy_to_host(bool val) { m_needs_copy_to_host = val; }
            bool needs_copy_to_host() const { return m_needs_copy_to_host; }
        private:
            bool m_needs_default_layout{false};
            bool m_needs_copy_to_host{true};
        };
    }
}
