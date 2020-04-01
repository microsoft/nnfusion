// Microsoft (c) 2019, NNFusion Team

#include "elementwise_arithmetic.hpp"
#include "nnfusion/core/graph/gnode.hpp"

#include "nnfusion/common/partial_shape.hpp"
#include "nnfusion/common/type/element_type.hpp"

using namespace nnfusion::op;

ElementwiseArithmetic::ElementwiseArithmetic(const std::string& node_type)
    : Op(node_type)
{
}

void ElementwiseArithmetic::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto args_et_pshape = validate_and_infer_elementwise_args(gnode);

    nnfusion::element::Type& args_et = std::get<0>(args_et_pshape);
    nnfusion::PartialShape& args_pshape = std::get<1>(args_et_pshape);

    OP_VALIDATION(this, args_et.is_dynamic() || args_et != nnfusion::element::boolean)
        << "Arguments cannot have boolean element type (argument element type: " << args_et << ").";

    gnode->set_output_type_and_shape(0, args_et, args_pshape);
}
