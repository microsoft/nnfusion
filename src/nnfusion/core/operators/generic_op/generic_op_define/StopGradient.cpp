// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(StopGradient).infershape(nnfusion::op::infershape::copy_shape_from_inputs);
