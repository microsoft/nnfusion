// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// Metagraph IR, which is to guide the codegen procedcure.
// This IR is based on ONNIX::ir's interface, but
// Instructions has attribute, namespace, and tag

#include "instruction.hpp"

using namespace nnfusion;
using namespace nnfusion::ir;
using namespace nnfusion::graph;

Instruction::Instruction(const nnfusion::graph::GNode& gnode)
{
    this->copy_tags_from(gnode);
}