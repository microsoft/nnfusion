// Microsoft (c) 2019, NNFusion Team

#include "reduce.hpp"
#include "nnfusion/common/util.hpp"
#include "nnfusion/core/graph/graph.hpp"

using namespace std;
using namespace nnfusion::op;

Reduce::Reduce(const shared_ptr<graph::Graph>& reduction_graph,
               const nnfusion::AxisSet& reduction_axes)
    : Op("Reduce")
    , m_reduction_graph(reduction_graph)
    , m_reduction_axes(reduction_axes)
{
}

void Reduce::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto input_reductee = gnode->get_inputs().at(0);

    auto input_init = gnode->get_inputs().at(1);
    CHECK(input_init->get_shape().size() == 0) << "Argument for initial value is not a scalar";

    CHECK(input_init->get_element_type() == input_reductee->get_element_type())
        << "Element types for reductee and initial values do not match";

    auto input_reductee_shape = input_reductee->get_shape();

    for (auto axis : m_reduction_axes)
    {
        CHECK(axis < input_reductee_shape.size()) << "Reduction axis is out of bounds";
    }

    nnfusion::Shape result_shape;

    for (size_t i = 0; i < input_reductee_shape.size(); i++)
    {
        if (m_reduction_axes.count(i) == 0)
        {
            result_shape.push_back(input_reductee_shape.at(i));
        }
    }

    auto g_params = m_reduction_graph->get_parameters();
    auto arg_reductee = gnode->get_in_edge(0)->get_src();
    auto arg_init = gnode->get_in_edge(1)->get_src();
    CHECK(g_params.size() == 2) << "Reduction graph has wrong number of parameters (should be two)";

    CHECK(g_params.at(0)->has_same_type(arg_init))
        << "Argument 0 of reduction graph has wrong type";
    CHECK(g_params.at(1)->has_same_type(arg_init))
        << "Argument 1 of reduction graph has wrong type";

    CHECK(m_reduction_graph->get_output_size() == 1) << "Single-output reduce graph was expected!";

    CHECK(m_reduction_graph->get_outputs().at(0)->get_element_type() ==
          arg_init->get_element_type())
        << "Return element type from reduction graph does not match expected";
    CHECK(m_reduction_graph->get_outputs().at(0)->get_shape() == nnfusion::Shape{})
        << "Return shape from reduction graph is not a scalar";
    gnode->set_output_type_and_shape(0, input_reductee->get_element_type(), result_shape);
}
