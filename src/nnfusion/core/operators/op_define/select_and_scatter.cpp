// Microsoft (c) 2019, NNFusion Team

#include "select_and_scatter.hpp"
#include "nnfusion/common/util.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"

using namespace std;
using namespace nnfusion::op;

SelectAndScatter::SelectAndScatter(const std::shared_ptr<graph::Graph>& selection_graph,
                                   const std::shared_ptr<graph::Graph>& scatter_graph,
                                   const nnfusion::Shape& window_shape,
                                   const nnfusion::Strides& window_movement_strides)
    : Op("SelectAndScatter")
    , m_selection_graph(selection_graph)
    , m_scatter_graph(scatter_graph)
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
{
}

void SelectAndScatter::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto input_selectee = gnode->get_inputs().at(0);
    auto input_source = gnode->get_inputs().at(1);
    auto input_init = gnode->get_inputs().at(2);
    auto input_selectee_shape = input_selectee->get_shape();
    auto input_source_shape = input_source->get_shape();
    auto input_init_shape = input_init->get_shape();
    auto input_selectee_element_type = input_selectee->get_element_type();
    auto input_source_element_type = input_source->get_element_type();
    auto input_init_element_type = input_init->get_element_type();

    //
    // Make sure the initial value is a scalar.
    //
    CHECK(input_init_shape.size() == 0) << "Argument for initial value is not a scalar";
    //
    // Make sure input element types all match.
    //
    CHECK(input_init_element_type == input_selectee_element_type)
        << "Element types for selectee and initial values do not match";
    CHECK(input_source_element_type == input_selectee_element_type)
        << "Element types for selectee and source tensors do not match";
    //
    // Check that the window shape and strides have the right rank.
    //
    CHECK(input_selectee_shape.size() == m_window_shape.size())
        << "Window shape has different rank from selectee tensor";
    CHECK(input_selectee_shape.size() == m_window_movement_strides.size())
        << "Window movement strides have different rank from selectee tensor";
    //
    // Check for zero-length window axes or strides.
    //
    for (size_t s : m_window_shape)
    {
        CHECK(s != 0) << "Window shape has a zero-length axis";
    }

    for (size_t s : m_window_movement_strides)
    {
        CHECK(s != 0) << "Window movement stride for some axis is zero";
    }

    //
    // Check that the window is not bigger than the selectee tensor.
    //
    for (size_t i = 0; i < input_selectee_shape.size(); i++)
    {
        CHECK(m_window_shape[i] <= input_selectee_shape[i])
            << "Reduction window is bigger than selectee tensor";
    }

    //
    // The expected shape of the source tensor is the same as the shape of the output
    // we would get if we window-reduced the selectee; in other words, this logic is
    // the same as the logic for computing the output shape of reduce-window.
    //
    nnfusion::Shape expected_source_shape;

    for (size_t i = 0; i < input_selectee_shape.size(); i++)
    {
        expected_source_shape.push_back(ceil_div(input_selectee_shape[i] - m_window_shape[i] + 1,
                                                 m_window_movement_strides[i]));
    }
    CHECK(input_source_shape == expected_source_shape)
        << "Source tensor does not have expected shape";
    //
    // Check the type signature of the selection graph. Should be T -> T -> Bool.
    //
    auto selection_graph_params = m_selection_graph->get_parameters();
    auto arg_reductee = gnode->get_in_edge(0)->get_src();
    auto arg_init = gnode->get_in_edge(1)->get_src();
    CHECK(selection_graph_params.size() == 2)
        << "Selection graph has wrong number of parameters (should be two)";
    CHECK(selection_graph_params.at(0)->get_element_type() == arg_init->get_element_type())
        << "Parameter 0 of selection graph has wrong element type";
    CHECK(selection_graph_params.at(1)->get_element_type() == arg_init->get_element_type())
        << "Parameter 1 of selection graph has wrong element type";
    CHECK(selection_graph_params.at(0)->get_shape() == nnfusion::Shape{})
        << "Parameter 0 of selection graph is not a scalar";
    CHECK(selection_graph_params.at(1)->get_shape() == nnfusion::Shape{})
        << "Parameter 1 of selection graph is not a scalar";
    CHECK(m_selection_graph->get_output_size() <= 1)
        << "Single-output selection graph was expected";
    CHECK(m_selection_graph->get_outputs().at(0)->get_element_type() == nnfusion::element::boolean)
        << "Return element type from selection graph is not boolean";
    CHECK(m_selection_graph->get_outputs().at(0)->get_shape() == nnfusion::Shape{})
        << "Return shape from selection graph is not a scalar";
    //
    // Check the type signature of the scatter graph. Should be T -> T -> T.
    //
    auto scatter_graph_params = m_scatter_graph->get_parameters();

    CHECK(scatter_graph_params.size() == 2)
        << "Scatter graph has wrong number of parameters (should be two)";
    CHECK(scatter_graph_params.at(0)->get_element_type() == arg_init->get_element_type())
        << "Parameter 0 of scatter graph has wrong element type";
    CHECK(scatter_graph_params.at(1)->get_element_type() == arg_init->get_element_type())
        << "Parameter 1 of scatter graph has wrong element type";
    CHECK(scatter_graph_params.at(0)->get_shape() == nnfusion::Shape{})
        << "Parameter 0 of scatter graph is not a scalar";
    CHECK(scatter_graph_params.at(1)->get_shape() == nnfusion::Shape{})
        << "Parameter 1 of scatter graph is not a scalar";
    CHECK(m_scatter_graph->get_output_size() <= 1) << "Single-output scatter graph was expected";
    CHECK(m_scatter_graph->get_outputs().at(0)->get_element_type() == arg_init->get_element_type())
        << "Return element type from scatter graph does not match the init value type";
    CHECK(m_scatter_graph->get_outputs().at(0)->get_shape() == nnfusion::Shape{})
        << "Return shape from scatter graph is not a scalar";
    //
    // Result type is the same element type and shape as the selectee.
    //
    gnode->set_output_type_and_shape(0, input_selectee_element_type, input_selectee_shape);
}
