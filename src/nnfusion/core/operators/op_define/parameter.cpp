// Microsoft (c) 2019, NNFusion Team

#include "parameter.hpp"
#include "nnfusion/core/graph/gnode.hpp"
using namespace std;
using namespace nnfusion::op;

Parameter::Parameter(const nnfusion::element::Type& element_type,
                     const nnfusion::PartialShape& pshape,
                     const bool cacheable)
    : Op("Parameter")
    , m_cacheable(cacheable)
    , m_partial_shape(pshape)
    , m_element_type(element_type)
{
}

void Parameter::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    Op::validate_and_infer_types(gnode);

    gnode->set_output_type_and_shape(0, m_element_type, m_partial_shape);
}
