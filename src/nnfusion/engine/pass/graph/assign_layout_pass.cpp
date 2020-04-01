// Microsoft (c) 2019, NNFusion Team

#include "assign_layout_pass.hpp"
#include "nnfusion/common/descriptor/layout/dense_tensor_layout.hpp"
#include "nnfusion/util/util.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool AssignLayoutPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    for (auto gnode : graph->get_nodes())
    {
        try
        {
            for (size_t i = 0; i < gnode->get_output_size(); ++i)
            {
                auto tv = gnode->get_output_tensor_ptr(i);
                if (nullptr == tv->get_tensor_layout())
                {
                    auto layout =
                        std::make_shared<nnfusion::descriptor::layout::DenseTensorLayout>(*tv);
                    tv->set_tensor_layout(layout);
                }
            }
        }
        catch (const std::exception& e)
        {
            CHECK_FAIL_WITH_EXCEPTION(errors::InvalidArgument)
                << "Error with node " << gnode->get_unique_name() << ": " << e.what();
        }
    }
    return true;
}
