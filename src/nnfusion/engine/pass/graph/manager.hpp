// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/core/graph/graph.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class GraphPassManager
            {
            public:
                GraphPassManager();
                ~GraphPassManager();

                void initialize_default_passes();

                template <typename T, class... Args>
                void register_pass(Args&&... args)
                {
                    static_assert(std::is_base_of<GraphPassBase, T>::value,
                                  "pass not derived from graph pass base");
                    auto pass = std::make_shared<T>(std::forward<Args>(args)...);
                    auto pass_base = std::static_pointer_cast<GraphPassBase>(pass);
                    m_pass_list.push_back(pass_base);
                    m_pass_names.push_back(typeid(T).name());
                }

                bool run_passes(std::shared_ptr<nnfusion::graph::Graph> graph);

            private:
                std::vector<std::string> m_pass_names;
                std::vector<std::shared_ptr<GraphPassBase>> m_pass_list;
            };
        } //namespace pass
    }     // namespace graph
} // namespace nnfusion