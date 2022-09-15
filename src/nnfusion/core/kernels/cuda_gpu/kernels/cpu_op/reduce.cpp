// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "reduce.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "Max",
    Device(SINGLE_CPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda_cpu::Reduce<nnfusion::op::Max>)

REGISTER_KERNEL_EMITTER(
    "Max",                                                                     // op_name
    Device(SINGLE_CPU).TypeConstraint(element::f32).Tag("cuda_lib").Priority(2), // attrs
    cuda_cpu::ReduceMemcpy<nnfusion::op::Max>)

REGISTER_KERNEL_EMITTER(
    "Min",
    Device(SINGLE_CPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda_cpu::Reduce<nnfusion::op::Min>)

REGISTER_KERNEL_EMITTER(
    "Min",                                                                     // op_name
    Device(SINGLE_CPU).TypeConstraint(element::f32).Tag("cuda_lib").Priority(2), // attrs
    cuda_cpu::ReduceMemcpy<nnfusion::op::Min>)

REGISTER_KERNEL_EMITTER(
    "Product",
    Device(SINGLE_CPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda_cpu::Reduce<nnfusion::op::Multiply>)

REGISTER_KERNEL_EMITTER(
    "Product",                                                                 // op_name
    Device(SINGLE_CPU).TypeConstraint(element::f32).Tag("cuda_lib").Priority(2), // attrs
    cuda_cpu::ReduceMemcpy<nnfusion::op::Multiply>)

REGISTER_KERNEL_EMITTER(
    "Sum",
    Device(SINGLE_CPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda_cpu::Reduce<nnfusion::op::Add>)

REGISTER_KERNEL_EMITTER(
    "Sum",                                                                     // op_name
    Device(SINGLE_CPU).TypeConstraint(element::f32).Tag("cuda_lib").Priority(2), // attrs
    cuda_cpu::ReduceMemcpy<nnfusion::op::Add>)

REGISTER_KERNEL_EMITTER(
    "ReduceAny",
    Device(SINGLE_CPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda_cpu::Reduce<nnfusion::op::Or>)

REGISTER_KERNEL_EMITTER(
    "ReduceAny",                                                               // op_name
    Device(SINGLE_CPU).TypeConstraint(element::f32).Tag("cuda_lib").Priority(2), // attrs
    cuda_cpu::ReduceMemcpy<nnfusion::op::Or>)