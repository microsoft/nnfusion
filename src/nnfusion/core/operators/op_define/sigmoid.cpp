// Microsoft (c) 2019, NNFusion Team

#include "sigmoid.hpp"

using namespace nnfusion::op;

Sigmoid::Sigmoid()
    : ElementwiseArithmetic("Sigmoid")
{
}

SigmoidBackprop::SigmoidBackprop()
    : ElementwiseArithmetic("SigmoidBackprop")
{
}
