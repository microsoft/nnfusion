#include "cpu_op_emitter.hpp"

using namespace nnfusion;
using namespace kernels;

cuda_cpu::CPUOpEmitter::CPUOpEmitter(std::shared_ptr<KernelContext> ctx) : KernelEmitter(ctx, "single_cpu") {}

LanguageUnit_p cuda_cpu::CPUOpEmitter::emit_function_call() {
    auto gnode = m_context->gnode;
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_call"));
    auto& lu = *_lu;
    vector<string> names;
    for (auto name: m_context->input_names) names.push_back(name + "_cpu");
    for (auto name: m_context->output_names) names.push_back(name + "_cpu");
    for (auto name: m_context->tensor_names) names.push_back(name + "_cpu");
    lu << "(" << join(names, ", ") << ");\n";
    return _lu;
}

LanguageUnit_p cuda_cpu::CPUOpEmitter::emit_function_body() {
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_body"));
    auto& lu = *_lu;
    lu << "// todo: " << m_context->op->get_unique_name();
    return _lu;
}

LanguageUnit_p cuda_cpu::CPUOpEmitter::emit_dependency() {
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    auto& lu = *_lu;
    lu << "// todo: dep";
    return _lu;
}