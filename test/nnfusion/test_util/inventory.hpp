// Microsoft (c) 2019, Wenxiang
/**
 * \brief Predefined Operator Inventory for Unit Tests
 * \author wenxh
 */
#pragma once

#include "nnfusion/core/graph/gnode.hpp"

#include <memory>
#include <vector>
using namespace std;

namespace nnfusion
{
    namespace inventory
    {
        template <class T, class dtype = float>
        shared_ptr<graph::GNode> create_object(int option = 0);
        template <class T, class dtype>
        vector<dtype> generate_input(int option = 0);
        template <class T, class dtype>
        vector<dtype> generate_output(int option = 0);
        template <class T, class dtype>
        vector<dtype> generate_param(int option = 0);
    }
}