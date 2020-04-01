// Microsoft (c) 2019, NNFusion Team

#include "reduce.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

#define REGISTER_GPU_KERNEL(OP_NAME)                                                               \
    REGISTER_KERNEL_EMITTER("Reduce",                                                              \
                            Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel" #OP_NAME), \
                            cuda::Reduce<nnfusion::op::OP_NAME>)

REGISTER_GPU_KERNEL(Max)
REGISTER_GPU_KERNEL(Min)

REGISTER_KERNEL_EMITTER("Sum",
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"),
                        cuda::Reduce<nnfusion::op::Add>)

REGISTER_KERNEL_EMITTER("Sum",                                                     // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_lib"), // attrs
                        cuda::ReduceMemcpy<nnfusion::op::Add>)

REGISTER_KERNEL_EMITTER("Sum",
                        Device(ROCM_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"),
                        cuda::Reduce<nnfusion::op::Add>)

REGISTER_KERNEL_EMITTER("Sum",                                                     // op_name
                        Device(ROCM_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_lib"), // attrs
                        cuda::ReduceMemcpy<nnfusion::op::Add>)
