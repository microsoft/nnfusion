// Microsoft (c) 2019, NNFusion Team

#include "divide.hpp"

using namespace nnfusion::op;

Divide::Divide()
    : ElementwiseArithmetic("Divide")
{
}

DivNoNan::DivNoNan()
    : ElementwiseArithmetic("DivNoNan")
{
}