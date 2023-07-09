// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/util/util.hpp"

using namespace nnfusion::kernels::cuda;

namespace nnfusion
{
    namespace blockfusion
    {
        const static std::vector<std::string> BlockFusionSupportBackend = {"CUDA", "ROCm"};

        class BlockExecutorInstruction
        {
        public:
            BlockExecutorInstruction(int _be_id)
                : be_id(_be_id)
            {
            }
            virtual ~BlockExecutorInstruction() = default;
            // inline int get_be_id() { return be_id; }
            // virtual void codegen() = 0;

        public:
            int be_id; // bind to which block executor
        };

        using BEInstruction_p = std::shared_ptr<BlockExecutorInstruction>;
        using BlockKernel_p = std::shared_ptr<BlockCudaEmitter>;

        class BlockExecutorProgram
        {
        public:
            size_t num_bes;
            std::vector<std::vector<BEInstruction_p>> block_executor_instructions;
            std::vector<BlockKernel_p> block_kernels; // (key, value): (kernel_id, BlockCudaEmitter)
        };

        using BEProgram_p = std::shared_ptr<BlockExecutorProgram>;

        class KernelMetric
        {
        public:
            double duration;
        };

        using KernelMetric_p = std::shared_ptr<KernelMetric>;

        class ProfilingResult
        {
        public:
            bool profile_device;  // whether profiled block_parallel_device
            bool profile_codegen; // whether profiled blockfusion_codegen

            // metrics for block_parallel_device
            size_t num_bes;               // number of BEs in this BlockFusion-pass
            size_t num_kernels;           // number of kernels in this BlockFusion-pass
            size_t num_large_kernels;     // large-kernel: grid_size >= num_bes
            double normal_execution_time; // execution time (us) without BlockFusion
            double fused_estimation_time; // estimated execution time (us) with BlockFusion

            // metrics for blockfusion_codegen
            size_t num_parameters;       // number of parameters in the fused kernel
            double fused_execution_time; // execution time (us) with BlockFusion

        public:
            std::string get_debug_string()
            {
                std::ostringstream ret;

                ret << "=====ProfilingResult for this group start=====\n";

                ret << "ProfilingResult.profile_device: " << std::to_string(profile_device) << "\n";
                ret << "ProfilingResult.num_bes: " << std::to_string(num_bes) << "\n";
                ret << "ProfilingResult.num_kernels: " << std::to_string(num_kernels) << "\n";
                ret << "ProfilingResult.num_large_kernels: " << std::to_string(num_large_kernels)
                    << "\n";
                ret << "ProfilingResult.normal_execution_time: "
                    << std::to_string(normal_execution_time) << "\n";
                ret << "ProfilingResult.fused_estimation_time: "
                    << std::to_string(fused_estimation_time) << "\n";
                ret << "ProfilingResult.profile_codegen: " << std::to_string(profile_codegen)
                    << "\n";
                ret << "ProfilingResult.num_parameters: " << std::to_string(num_parameters) << "\n";
                ret << "ProfilingResult.fused_execution_time: "
                    << std::to_string(fused_execution_time) << "\n";

                ret << "=====ProfilingResult for this group end=====\n";

                return ret.str();
            }
        };

        using ProfilingResult_p = std::shared_ptr<ProfilingResult>;

        /* ===== different types of block_executor_instructions begin ===== */

        class BlockExecutorInstructionExecuteBlock : public BlockExecutorInstruction
        {
        public:
            BlockExecutorInstructionExecuteBlock(int _be_id, int _kernel_id, int _kernel_block_id)
                : BlockExecutorInstruction(_be_id)
                , kernel_id(_kernel_id)
                , kernel_block_id(_kernel_block_id)
            {
            }
            // inline int get_kernel_id() { return kernel_id; }
            // inline int get_kernel_block_id() { return kernel_block_id; }

        public:
            int kernel_id;       // execute which kernel
            int kernel_block_id; // execute which block of a kernel
        };

        class BlockExecutorInstructionWaitFor : public BlockExecutorInstruction
        {
        public:
            BlockExecutorInstructionWaitFor(int _be_id,
                                            const std::vector<int>& _bes_predecessor,
                                            int _step_id)
                : BlockExecutorInstruction(_be_id)
            {
                bes_predecessor = _bes_predecessor;
                step_id = _step_id;
            }
            // inline std::vector<int> get_bes_predecessor() { return bes_predecessor; }
            // inline get_step_id() { return step_id; }

        public:
            std::vector<int> bes_predecessor; // wait which bes, these bes have the same step
            int step_id;                      // wait for which step
        };

        class BlockExecutorInstructionStepTo : public BlockExecutorInstruction
        {
        public:
            BlockExecutorInstructionStepTo(int _be_id, int _step_id)
                : BlockExecutorInstruction(_be_id)
                , step_id(_step_id)
            {
            }
            // inline get_step_id() { return step_id; }

        public:
            int step_id; // sync to step, note that bes should step to the same step
        };

        /* ===== different types of block_executor_instructions end ===== */
    } // namespace blockfusion
} // namespace nnfusion