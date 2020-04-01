// Microsoft (c) 2019, NNFusion Team

#include "relu.hpp"

using namespace std;
using namespace nnfusion::op;

Relu::Relu()
    : ElementwiseArithmetic("Relu")
{
}

ReluBackprop::ReluBackprop()
    : ElementwiseArithmetic("ReluBackprop")
{
}