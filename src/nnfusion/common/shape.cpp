//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "nnfusion/common/shape.hpp"
#include "nnfusion/common/symbolic_shape.hpp"
#include "nnfusion/common/util.hpp"

std::ostream& nnfusion::operator<<(std::ostream& s, const Shape& shape)
{
    s << "Shape{";
    s << nnfusion::join(shape);
    s << "}";
    if (shape.is_dynamic())
    {
        s << " SymShape: [" << (*shape.get_sym_shape()) << "]";
    }
    return s;
}

bool Shape::is_dynamic() const
{
    return (sym_shape && sym_shape->is_dynamic());
}

bool is_shape_compatible(const Shape& shape, const SymShape& sym_shape)
{
    if (shape.size() != sym_shape.size())
    {
        return false;
    }
    for (size_t i = 0; i < shape.size(); i++)
    {
        size_t dim_i = shape[i];
        SymDim symdim_i = sym_shape[i];
        if (symdim_i.is_static())
        {
            if (dim_i != symdim_i.max())
            {
                return false;
            }
        }
        else
        {
            if (dim_i < symdim_i.min() || dim_i > symdim_i.max())
            {
                return false;
            }
        }
    }
    return true;
}

void Shape::set_sym_shape(std::shared_ptr<SymShape> shape)
{
    NNFUSION_CHECK(is_shape_compatible(*this, *shape)) << "Incompatible sym shape: " << (*shape)
                                                       << ", origin shape: " << (*this);
    sym_shape = shape;
}
