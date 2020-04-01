// Microsoft (c) 2019, NNFusion Team

#include "min.hpp"

using namespace nnfusion::op;

Min::Min(const nnfusion::AxisSet& reduction_axes)
    : ArithmeticReduction("Min", reduction_axes)
{
}
