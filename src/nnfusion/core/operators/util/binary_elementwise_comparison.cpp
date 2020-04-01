// Microsoft (c) 2019, NNFusion Team

#include "binary_elementwise_comparison.hpp"
#include "nnfusion/core/graph/gnode.hpp"

#include "nnfusion/common/partial_shape.hpp"
#include "nnfusion/common/type/element_type.hpp"

using namespace nnfusion::op;

BinaryElementwiseComparison::BinaryElementwiseComparison(const std::string& node_type)
    : Op(node_type)
{
}

void BinaryElementwiseComparison::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto args_et_pshape = validate_and_infer_elementwise_args(gnode);
    nnfusion::PartialShape& args_pshape = std::get<1>(args_et_pshape);

    gnode->set_output_type_and_shape(0, nnfusion::element::boolean, args_pshape);
}
