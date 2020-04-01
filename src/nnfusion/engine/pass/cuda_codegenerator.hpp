// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_langunit.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/engine/interpreter.hpp"
#include "nnfusion/engine/op.hpp"

using namespace nnfusion::kernels;
namespace nnfusion
{
    using KernelContext = nnfusion::kernels::KernelContext;

    class CudaCodeGenerator : public IInterpreterPass
    {
    public:
        bool run(std::shared_ptr<InterpreterContext> ctx,
                 std::shared_ptr<TranslationUnit> tu) override;

        virtual std::string get_generate_cmakelists(void);
        virtual void post_projgen(void);
        virtual void after_projgen(void);
        virtual std::string get_target_name(void);
        virtual std::vector<shared_ptr<const kernels::KernelRegistration>>
            find_backend_kernels(const std::string& op_name, const shared_ptr<KernelContext>& ctx);
        virtual KernelEmitter::Pointer
            match_kernel(std::vector<pair<DeviceType, KernelEmitter::Pointer>>& res);
        virtual DeviceType device_type() { return DeviceType::CUDA_GPU; }
    private:
        virtual bool projgen();
        virtual bool setpwd();

    protected:
        LanguageUnit_p lu_cmakefile, lu_nnfusion_rt, lu_header, lu_main;
        bool super_scaler_enable = false;
    };
} // namespace nnfusion
