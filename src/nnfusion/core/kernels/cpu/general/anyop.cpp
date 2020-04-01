// Microsoft (c) 2019, NNFusion Team

#include "anyop.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cpu::AnyOP::AnyOP(shared_ptr<KernelContext> ctx)
    : CpuKernelEmitter(ctx)
{
}

LanguageUnit_p cpu::AnyOP::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    // function signature:
    // void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
    lu.block_begin();
    {
        lu << "// This function is left empty by purpose.\n";
    }
    lu.block_end();
    return _lu;
}

LanguageUnit_p cpu::AnyOP::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

    return _lu;
}

// Register Pad kernel emitter

REGISTER_KERNEL_EMITTER("AnyOP",                                      //op_name
                        Device(GENERIC_CPU).TypeConstraint(DT_FLOAT), //attrs
                        cpu::AnyOP)                                   // constructor