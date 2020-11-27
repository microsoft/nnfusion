// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "reduce.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "Max",
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda::Reduce<nnfusion::op::Max>)

REGISTER_KERNEL_EMITTER(
    "Max",                                                                     // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_lib").Priority(2), // attrs
    cuda::ReduceMemcpy<nnfusion::op::Max>)

REGISTER_KERNEL_EMITTER(
    "Min",
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda::Reduce<nnfusion::op::Min>)

REGISTER_KERNEL_EMITTER(
    "Min",                                                                     // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_lib").Priority(2), // attrs
    cuda::ReduceMemcpy<nnfusion::op::Min>)

REGISTER_KERNEL_EMITTER(
    "Product",
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda::Reduce<nnfusion::op::Multiply>)

REGISTER_KERNEL_EMITTER(
    "Product",                                                                 // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_lib").Priority(2), // attrs
    cuda::ReduceMemcpy<nnfusion::op::Multiply>)

REGISTER_KERNEL_EMITTER(
    "Sum",
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda::Reduce<nnfusion::op::Add>)

REGISTER_KERNEL_EMITTER(
    "Sum",                                                                     // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_lib").Priority(2), // attrs
    cuda::ReduceMemcpy<nnfusion::op::Add>)

REGISTER_KERNEL_EMITTER(
    "Sum",
    Device(ROCM_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda::Reduce<nnfusion::op::Add>)

REGISTER_KERNEL_EMITTER(
    "Sum",                                                                     // op_name
    Device(ROCM_GPU).TypeConstraint(element::f32).Tag("cuda_lib").Priority(2), // attrs
    cuda::ReduceMemcpy<nnfusion::op::Add>)

REGISTER_KERNEL_EMITTER(
    "ReduceAny",
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda::Reduce<nnfusion::op::Or>)

REGISTER_KERNEL_EMITTER(
    "ReduceAny",                                                               // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_lib").Priority(2), // attrs
    cuda::ReduceMemcpy<nnfusion::op::Or>)