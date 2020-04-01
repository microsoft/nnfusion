// Microsoft (c) 2019, NNFusion Team

#include "product.hpp"

using namespace nnfusion::op;

Product::Product(const nnfusion::AxisSet& reduction_axes)
    : ArithmeticReduction("Product", reduction_axes)
{
}
