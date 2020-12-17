// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "kernel_tuning.hpp"
#include <queue>
#include <sstream>
#include <utility>

#include "nnfusion/core/kernels/cpu/cpu_kernel_emitter.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/core/kernels/hlsl/hlsl_kernel_emitter.hpp"
#include "nnfusion/util/curl_request.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::kernels;
using namespace nnfusion::pass::graph;

DEFINE_int64(fkernel_tuning_steps, 0, "Enable automatic kernel tuning for maximum N steps.");
DECLARE_bool(fantares_mode);
DECLARE_string(fantares_codegen_server);
DECLARE_string(fproduct_name);

const std::unordered_set<std::string> KernelTuning::BlockList = {};

struct TuningStatus
{
    TuningStatus(std::shared_ptr<GNode> gnode)
        : op_type(gnode->get_op_type())
        , op_name(gnode->get_op_ptr()->get_name())
        , progress_step(0)
        , best_perf(-1.0)
    {
    }
    std::string op_type;
    std::string op_name;
    std::string status;
    int64_t progress_step;
    double best_perf;
    std::string ir;
};

std::string send_tuning_request(std::string& ir, int64_t step)
{
    CurlRequest req(FLAGS_fantares_codegen_server);
    req.add_custom_header(("COMPUTE_V1: " + ir).c_str());
    req.add_custom_header(("STEP: " + std::to_string(step)).c_str());

    std::string response;
    if (!req.send_request(response))
    {
        NNFUSION_LOG(ERROR) << "Error request for IR: " << ir;
    }
    // NNFUSION_LOG(INFO) << response;
    if (strncmp(response.c_str(), "[ERROR]", 7) == 0)
    {
        NNFUSION_LOG(ERROR) << ir << "\n" << response;
    }

    return response;
}

void print_tuning_results(std::vector<std::shared_ptr<TuningStatus>> tuned_kernels,
                          std::vector<std::shared_ptr<TuningStatus>> tuning_kernels)
{
    std::stringstream ss;
    ss << " Kernel Tuning Status: \n";
    ss << " NOTE: the tuning progress (N/M) means that the current best kernel is searched at the "
          "N-th step of the total M steps. \n\n";
    ss << " | " << std::setw(20) << "OP"
       << " | " << std::setw(26) << "NAME"
       << " | " << std::setw(10) << "STATUS"
       << " | " << std::setw(10) << "PROGRESS"
       << " | " << std::setw(18) << "PERFORMANCE |\n";

    ss << " | " << std::setfill('-') << setw(96) << " |\n";
    ss << std::setfill(' ');

    for (auto s : tuned_kernels)
    {
        ss << " | " << std::setw(20) << s->op_type << " | " << std::setw(26)
           << ((s->op_name.size() > 26) ? (s->op_name.substr(0, 24) + "..") : s->op_name) << " | "
           << std::setw(10) << s->status << " | " << std::setw(6) << s->progress_step << "/"
           << FLAGS_fkernel_tuning_steps << " "
           << " | " << std::setw(12) << s->best_perf << " ms |\n";
    }
    for (auto s : tuning_kernels)
    {
        ss << " | " << std::setw(20) << s->op_type << " | " << std::setw(26)
           << ((s->op_name.size() > 26) ? (s->op_name.substr(0, 24) + "..") : s->op_name) << " | "
           << std::setw(10) << s->status << " | " << std::setw(6) << s->progress_step << "/"
           << FLAGS_fkernel_tuning_steps << " "
           << " | " << std::setw(12) << s->best_perf << " ms |\n";
    }
    NNFUSION_LOG(INFO) << ss.str();
}

std::vector<std::shared_ptr<GNode>>
    KernelTuning::get_tuning_candidates(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    NNFUSION_CHECK(graph != nullptr);

    std::vector<std::shared_ptr<GNode>> candidates;
    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();

    std::unordered_set<std::string> translated_irs;
    for (auto gnode : nodes)
    {
        if (!(*gnode)["DeviceType"].is_valid())
        {
            NNFUSION_CHECK_FAIL() << "GNode DeviceType should be assigned before this pass："
                                  << gnode->get_name();
        }
        auto n_device_type = (*gnode)["DeviceType"].as<NNFusion_DeviceType>();
        NNFUSION_CHECK(n_device_type != UNKNOWN);

        // filter ops in BlockList
        if (BlockList.find(gnode->get_op_type()) != BlockList.end())
        {
            continue;
        }

        auto ir = nnfusion::op::get_translation(gnode);
        // NNFUSION_LOG(DEBUG) << gnode->get_op_type() << ", ir: " << ir;

        // filter unimplemented irs
        if (ir.empty())
        {
            continue;
        }

        // dedupe ops with the same ir
        if (!ir.empty())
        {
            if (translated_irs.find(ir) != translated_irs.end())
            {
                continue;
            }
            translated_irs.insert(ir);
        }

        candidates.push_back(gnode);
    }

    // filter ops existing in kernel cache DB
    {
        auto cache_manager = std::make_shared<cache::KernelCacheManager>();
        if (!cache_manager->is_valid())
        {
            NNFUSION_LOG(INFO) << "No valid kernel cache, all the kernels will be tuned";
        }
        else
        {
            std::vector<std::shared_ptr<GNode>> non_cached_candidates;
            for (auto gnode : candidates)
            {
                shared_ptr<KernelContext> ctx(new KernelContext(gnode));
                auto identifier = ctx->generate_identifier();
                auto device_type = get_device_str((*gnode)["DeviceType"].as<NNFusion_DeviceType>());
                std::string source = "Antares";
                auto fetched = cache_manager->fetch_with_source(identifier, device_type, source);

                bool tune_flag = true;
                for (auto fetch : fetched)
                {
                    if (fetch->miscs["antares"]["device_name"] == FLAGS_fproduct_name &&
                        fetch->miscs["antares"]["planned_steps"] >= FLAGS_fkernel_tuning_steps)
                    {
                        tune_flag = false;
                    }
                }
                if (tune_flag)
                {
                    non_cached_candidates.push_back(gnode);
                }
            }
            candidates = non_cached_candidates;
        }
    }

    return candidates;
}

bool KernelTuning::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (FLAGS_fantares_mode)
    {
        // register antares kernels anyway here in case kernel selection pass will use them
        register_antares_kernel();
    }

    if (FLAGS_fkernel_tuning_steps <= 0 || FLAGS_fantares_codegen_server == "" ||
        !FLAGS_fantares_mode)
    {
        return true;
    }

    std::vector<std::shared_ptr<TuningStatus>> tuned_kernels;
    std::vector<std::shared_ptr<TuningStatus>> tuning_kernels;

    std::vector<std::shared_ptr<GNode>> nodes = get_tuning_candidates(graph);
    for (auto gnode : nodes)
    {
        if (!(*gnode)["DeviceType"].is_valid())
        {
            NNFUSION_CHECK_FAIL() << "GNode DeviceType should be assigned before this pass："
                                  << gnode->get_name();
        }
        auto n_device_type = (*gnode)["DeviceType"].as<NNFusion_DeviceType>();
        NNFUSION_CHECK(n_device_type != UNKNOWN);

        auto ir = nnfusion::op::get_translation(gnode);
        // NNFUSION_LOG(INFO) << gnode->get_op_type() << ", ir: " << ir;
        if (!ir.empty())
        {
            auto status = std::make_shared<TuningStatus>(gnode);
            status->ir = ir;
            auto response = send_tuning_request(ir, 0);

            auto start = response.find("\n// Saved Perf =");
            if (start != std::string::npos)
            {
                auto tail_info = response.substr(start, string::npos);
                std::regex ws_re("\\s+");
                std::vector<std::string> items(
                    std::sregex_token_iterator(tail_info.begin(), tail_info.end(), ws_re, -1),
                    std::sregex_token_iterator());
                NNFUSION_CHECK(items.size() >= 16);

                double perf = std::stod(items[5]);
                int64_t best_step = std::stol(items[12]);
                int64_t plan_step = std::stol(items[16]);
                // NNFUSION_LOG(INFO) << "best perf: " << perf << "s, step: " << best_step;
                status->progress_step = best_step;
                status->best_perf = (perf < 0) ? -1 : perf * 1000.0;

                if (plan_step >= FLAGS_fkernel_tuning_steps)
                {
                    // no need to submit new tuning job
                    auto compelete_flag = response.find("Antares Tuning Completed in ");
                    status->status = (compelete_flag == string::npos) ? "tuning" : "completed";
                }
            }

            if (status->status == "" || status->status.empty())
            {
                // submit a new tuning task
                NNFUSION_LOG(INFO) << gnode->get_op_type() << ", ir: " << ir;
                auto response = send_tuning_request(ir, FLAGS_fkernel_tuning_steps);
                status->status = "submitted";
            }

            status->status == "completed" ? tuned_kernels.push_back(status)
                                          : tuning_kernels.push_back(status);
        }
    }
    print_tuning_results(tuned_kernels, tuning_kernels);

    if (tuning_kernels.size() > 0)
    {
        NNFUSION_LOG(NNFUSION_WARNING)
            << "There are pending tuning kernels. Please retry the compilation later!";
        exit(0);
    }

    insert_to_kernel_cache(nodes);

    return true;
}

bool KernelTuning::register_antares_kernel()
{
    for (auto pair : nnfusion::op::get_op_configs())
    {
        std::string op_name = pair.first;
        std::vector<NNFusion_DeviceType> devs{CUDA_GPU, GENERIC_CPU, HLSL};

        // skip op in BlockList
        if (BlockList.find(op_name) != BlockList.end())
        {
            continue;
        }

        kernels::KernelRegistrar kernel_registrar_cuda(
            op_name,
            kernels::Name(op_name)
                .Device(CUDA_GPU)
                .TypeConstraint(element::f32)
                .Tag("antares")
                .Priority(9)
                .KernelFactory([](shared_ptr<kernels::KernelContext> context)
                                   -> shared_ptr<kernels::KernelEmitter> {
                    return make_shared<kernels::cuda::AntaresCudaKernelEmitter>(context);
                })
                .Build());
        kernels::KernelRegistrar kernel_registrar_cpu(
            op_name,
            kernels::Name(op_name)
                .Device(GENERIC_CPU)
                .TypeConstraint(element::f32)
                .Tag("antares")
                .Priority(9)
                .KernelFactory([](shared_ptr<kernels::KernelContext> context)
                                   -> shared_ptr<kernels::KernelEmitter> {
                    return make_shared<kernels::cpu::AntaresCpuKernelEmitter>(context);
                })
                .Build());
        kernels::KernelRegistrar kernel_registrar_hlsl(
            op_name,
            kernels::Name(op_name)
                .Device(HLSL)
                .TypeConstraint(element::f32)
                .Tag("antares")
                .Priority(9)
                .KernelFactory([](shared_ptr<kernels::KernelContext> context)
                                   -> shared_ptr<kernels::KernelEmitter> {
                    return make_shared<kernels::hlsl::AntaresHLSLKernelEmitter>(context);
                })
                .Build());
    }
    return true;
}

bool KernelTuning::insert_to_kernel_cache(const std::vector<std::shared_ptr<GNode>>& nodes)
{
    auto cache_manager = std::make_shared<cache::KernelCacheManager>();
    if (!cache_manager->is_valid())
    {
        NNFUSION_LOG(INFO)
            << "No valid kernel cache, tuned kernels will not be inserted to kernel cache DB";
        return true;
    }

    for (auto gnode : nodes)
    {
        shared_ptr<KernelContext> ctx(new KernelContext(gnode));
        std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
            KernelRegistry::Global()->FindKernelRegistrations(
                gnode->get_op_type(), CUDA_GPU, element::f32);

        for (auto kernel_reg : kernel_regs)
        {
            if (kernel_reg->m_tag == "antares")
            {
                auto antares_kernel = kernel_reg->m_factory(ctx);
                auto kernel_cache_entry = antares_kernel->get_kernel_cache_entry();
                if (kernel_cache_entry == nullptr)
                {
                    NNFUSION_LOG(INFO)
                        << "Invalid kernel_cache_entry, will not insert to kernel cache: "
                        << gnode->get_name();
                }

                // overwrite existing kernel entries
                cache_manager->insert_kernel_entry(kernel_cache_entry, true);
            }
        }
    }
    return true;
}
