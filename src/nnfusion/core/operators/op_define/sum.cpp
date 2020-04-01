// Microsoft (c) 2019, NNFusion Team

#include "sum.hpp"

using namespace std;
using namespace nnfusion::op;

Sum::Sum(const nnfusion::AxisSet& reduction_axes)
    : ArithmeticReduction("Sum", reduction_axes)
{
}
