// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Pack).attr<int>("axis", 0).infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
    CHECK(gnode->get_input_size() >= 1) << "Inputs of Pack operator should not be less than 1.";

    auto& input_shape_0 = gnode->get_input_shape(0);
    for (int i = 1; i < gnode->get_input_size(); i++)
    {
        auto& input_shape_n = gnode->get_input_shape(i);
        CHECK(input_shape_0.size() == input_shape_n.size()) << "Shape dimension size not match.";
        for (int j = 0; j < input_shape_0.size(); j++)
        {
            CHECK(input_shape_0[j] == input_shape_n[j]) << "Dimension " << j
                                                        << " in shapes must be equal.";
        }
    }
    auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
    int axis = generic_op->localOpConfig.getRoot()["axis"];
    CHECK(axis >= 0 && axis <= input_shape_0.size())
        << "Pack dim " << axis << " not in the interval [" << 0 << ", " << input_shape_0.size()
        << "].";

    auto output_shape_0 = input_shape_0;
    output_shape_0.insert(output_shape_0.begin() + size_t(axis), gnode->get_input_size());

    gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
});
