// Microsoft (c) 2019, NNFusion Team

#include "max.hpp"

using namespace nnfusion::op;

Max::Max(const nnfusion::AxisSet& reduction_axes)
    : ArithmeticReduction("Max", reduction_axes)
{
}
