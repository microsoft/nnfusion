// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/async_manager.hpp"
#include "nnfusion/engine/op.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class AssignAsyncInfoPass : public GraphPassBase
            {
            public:
                AssignAsyncInfoPass();
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;

            private:
                void assign_stream_info(nnfusion::async::AsyncManager* async_manager,
                                        std::shared_ptr<Graph>& graph);
                void assign_event_info(nnfusion::async::AsyncManager* async_manager,
                                       std::shared_ptr<Graph>& graph);
                DeviceType m_device;
            };
        }
    }
}