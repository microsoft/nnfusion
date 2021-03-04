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

// Microsoft (c) 2019, NNFusion Team

#include "../op.hpp"

namespace nnfusion
{
    namespace op
    {
        class Fused : public Op
        {
        public:
            Fused(const std::string& name, const std::string& opname)
                : Op(opname){};

            void register_ir2(std::vector<std::shared_ptr<graph::GNode>>& gnodes);
            std::string get_fused_ir2() { return fused_op_ir2; };
            std::string get_plan_rule();

        protected:
            void assemble_inputs_and_outputs();

            std::string fused_op_ir2;
            std::vector<std::string> plan_rules;
        };
    }
}
