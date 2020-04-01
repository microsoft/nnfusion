// Microsoft (c) 2019, NNFusion Team

#include "softmax.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace nnfusion::op;

Softmax::Softmax(const nnfusion::AxisSet& axes)
    : ElementwiseArithmetic("Softmax")
    , m_axes(axes)
{
}

void Softmax::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    ElementwiseArithmetic::validate_and_infer_types(gnode);

    auto shape = gnode->get_output_shape(0);

    for (auto axis : m_axes)
    {
        OP_VALIDATION(this, axis < shape.size()) << "Reduction axis (" << axis
                                                 << ") is out of bounds (argument shape: " << shape
                                                 << ").";
    }

    // empty axes == all axes
    if (m_axes.size() == 0)
    {
        for (size_t i = 0; i < shape.size(); ++i)
        {
            m_axes.insert(i);
        }
    }
}
