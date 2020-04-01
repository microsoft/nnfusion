// Microsoft (c) 2019, NNFusion Team
#include "device_dispatcher.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool DefaultDeviceDispatcher::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    DeviceType dt = DeviceType::CUDA_GPU;
    /* for debug purpose
    switch(default_device)
    {
        case GENERIC_CPU:
        LOG(INFO) << "GENERIC_CPU";
        break;
        case  ROCM_GPU:
        LOG(INFO) << "ROCM_GPU";
        break;
        case CUDA_GPU:
        LOG(INFO) << "CUDA_GPU";
    }
    */

    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
    for (auto it : nodes)
    {
        it->Set<DeviceType>("Device", move(dt));
    }

    return true;
}