// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "kernel_tuning.hpp"
#include <queue>
#include <sstream>
#include <utility>

#include "nnfusion/common/util.hpp"
#include "nnfusion/core/kernels/cpu/cpu_kernel_emitter.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/core/kernels/hlsl/hlsl_kernel_emitter.hpp"
#include "nnfusion/util/curl_request.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::kernels;
using namespace nnfusion::pass::graph;

DEFINE_int64(fkernel_tuning_steps, 0, "Enable automatic kernel tuning for maximum N steps.");
DEFINE_int64(fdump_and_tune_irs, 0, "1 for dump irs, and 2 for load irs to tune");
DEFINE_double(fretuning_bar,
              0.0,
              "retry the tuning if existing kernel latency is higher than the bar (ms)");
DEFINE_string(ftuning_blocklist,
              "",
              "List of op types that skip kernel tuning pass, e.g., \"Softmax,Add\"");
DEFINE_string(
    ftuning_allowlist,
    "",
    "List of op types to tune kernel, e.g., \"Softmax,Add\", this will ignore the blocklist");
DEFINE_string(fantares_perf_file, "./antares_perf.csv", "File to save Antares kernel performance.");
DEFINE_string(ftuning_platform, "", "Antares platform: e.g., win64, xbox, etc.");
DEFINE_string(ftuning_agent, "", "Antares tuning agent ip address");
DECLARE_bool(fantares_mode);
DECLARE_string(fantares_codegen_server);
DECLARE_string(fproduct_name);
DECLARE_string(fdefault_device);
DECLARE_bool(fsymbolic);

std::string KernelTuning::send_tuning_request(std::string& ir, int64_t step, bool symbolic)
{
    auto server = symbolic ? m_dynamic_tuning_server : m_static_tuning_server;
    CurlRequest req(server);
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

void dump_perf(std::string filename,
               std::vector<std::shared_ptr<TuningStatus>> tuned_kernels,
               std::unordered_map<std::string, size_t> ir_cnt)
{
    double total_time = 0.0;
    for (auto status : tuned_kernels)
    {
        if (status->best_perf > 0)
            total_time += status->best_perf * ir_cnt.at(status->ir);
    }
    std::sort(tuned_kernels.begin(),
              tuned_kernels.end(),
              [&](std::shared_ptr<TuningStatus> left, std::shared_ptr<TuningStatus> right) {
                  return left->best_perf * ir_cnt.at(left->ir) >
                         right->best_perf * ir_cnt.at(right->ir);
              });
    std::ofstream out(FLAGS_fantares_perf_file);
    for (auto status : tuned_kernels)
    {
        size_t cnt = ir_cnt.at(status->ir);
        double kernel_time_sum = status->best_perf > 0 ? status->best_perf * cnt : -1.0;
        double percent = std::max(kernel_time_sum / total_time * 100, 0.0);

        out << std::fixed << std::setprecision(2) << percent << "%"
            << "\t" << kernel_time_sum << "\t" << cnt << "\t" << status->best_perf << "\t"
            << status->op_type << "\t" << status->op_name << "\t" << status->ir << endl;
    }
    out.close();
}

void dump_tuning_irs(std::string filename,
                     std::vector<std::shared_ptr<GNode>>& nodes,
                     std::unordered_map<std::string, size_t> ir_cnt)
{
    std::ofstream out(filename);
    for (auto gnode : nodes)
    {
        auto op = gnode->get_op_type();
        auto name = gnode->get_op_ptr()->get_name();
        bool symbolic = (FLAGS_fsymbolic && (*gnode)["symbolic"].is_valid_as<bool>());
        auto ir = nnfusion::op::get_translation(gnode);
        size_t cnt = ir_cnt.at(ir);
        out << op << "|" << name << "|" << symbolic << "|" << cnt << "|" << ir << endl;
    }
    out.close();
    NNFUSION_LOG(INFO) << "Dump all IRs into file: " << filename;
}

std::pair<std::vector<std::shared_ptr<GNode>>, std::vector<std::shared_ptr<TuningStatus>>>
    get_tuning_candidates(std::shared_ptr<nnfusion::graph::Graph>& graph,
                          const std::unordered_set<std::string> allow_list,
                          const std::unordered_set<std::string> block_list,
                          std::unordered_map<std::string, size_t>& ir2cnt)
{
    NNFUSION_CHECK(graph != nullptr);

    std::vector<std::shared_ptr<GNode>> candidates;
    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
    for (auto gnode : nodes)
    {
        if (!(*gnode)["DeviceType"].is_valid())
        {
            NNFUSION_CHECK_FAIL() << "GNode DeviceType should be assigned before this pass："
                                  << gnode->get_name();
        }
        auto n_device_type = (*gnode)["DeviceType"].as<NNFusion_DeviceType>();
        NNFUSION_CHECK(n_device_type != UNKNOWN);

        // only tune ops in AllowList
        if (allow_list.size() > 0 && allow_list.find(gnode->get_op_type()) == allow_list.end())
        {
            continue;
        }
        // filter ops in BlockList
        if (allow_list.size() == 0 && block_list.find(gnode->get_op_type()) != block_list.end())
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
        if (ir2cnt.find(ir) != ir2cnt.end())
        {
            ir2cnt.at(ir) += 1;
        }
        else
        {
            ir2cnt[ir] = 1;
            candidates.push_back(gnode);
        }
    }

    // filter ops existing in kernel cache DB
    std::vector<std::shared_ptr<TuningStatus>> cached_kernels;
    {
        auto cache_manager = std::make_shared<cache::KernelCacheManager>();
        if (!cache_manager->is_valid())
        {
            NNFUSION_LOG(INFO) << "No valid kernel cache, all the kernels will be tuned";
        }
        else
        {
            std::unordered_map<std::string, std::shared_ptr<TuningStatus>> ir2kernel;
            std::vector<std::shared_ptr<GNode>> non_cached_candidates;
            for (auto gnode : candidates)
            {
                auto ir = nnfusion::op::get_translation(gnode);
                shared_ptr<KernelContext> ctx(new KernelContext(gnode));
                auto identifier = ctx->generate_identifier();
                auto device_type = get_device_str((*gnode)["DeviceType"].as<NNFusion_DeviceType>());
                std::string source = "Antares";
                auto fetched = cache_manager->fetch_with_source(identifier, device_type, source);
                if (device_type != "CUDA_GPU")
                {
                    NNFUSION_CHECK(fetched.size() == 0);
                }

                bool tune_flag = true;
                for (auto fetch : fetched)
                {
                    if (fetch->miscs["antares"]["device_name"] == FLAGS_fproduct_name &&
                        fetch->miscs["antares"]["planned_steps"] >= FLAGS_fkernel_tuning_steps)
                    {
                        double fetch_perf = double(fetch->miscs["antares"]["time"]) / 1000;
                        // ignore kernel without perf
                        if (fetch_perf <= 0)
                        {
                            continue;
                        }
                        // ignore current kernel if we have a better kernel
                        if (ir2kernel.find(ir) != ir2kernel.end() &&
                            ir2kernel.at(ir)->best_perf <= fetch_perf)
                        {
                            continue;
                        }
                        auto status = std::make_shared<TuningStatus>(gnode);
                        status->status = "completed";
                        status->progress_step = fetch->miscs["antares"]["step_produced"];
                        status->best_perf = fetch_perf;
                        status->ir = ir;
                        ir2kernel[ir] = status;
                        tune_flag = false;
                    }
                }
                if (tune_flag)
                {
                    non_cached_candidates.push_back(gnode);
                }
                else
                {
                    cached_kernels.push_back(ir2kernel.at(ir));
                }
            }
            candidates = non_cached_candidates;
        }
    }

    return std::make_pair(candidates, cached_kernels);
}

bool KernelTuning::parse_allow_and_block_list()
{
    auto blocklist_str = FLAGS_ftuning_blocklist;
    auto allowlist_str = FLAGS_ftuning_allowlist;

    if (allowlist_str.size() > 0)
    {
        stringstream ss(allowlist_str);
        while (ss.good())
        {
            string substr;
            getline(ss, substr, ',');
            m_allow_list.insert(substr);
        }
        if (m_allow_list.size() > 0)
        {
            NNFUSION_LOG(INFO) << "Kernel Tuning AllowList: " << join(m_allow_list, ", ");
            return true;
        }
    }

    stringstream ss(blocklist_str);
    while (ss.good())
    {
        string substr;
        getline(ss, substr, ',');
        m_block_list.insert(substr);
    }
    NNFUSION_LOG(INFO) << "Kernel Tuning BlockList: " << join(m_block_list, ", ");
    return true;
}

void extract_tunning_status_from_kernel(std::string code, std::shared_ptr<TuningStatus> status)
{
    auto start = code.find("\n// Saved Perf =");
    if (start != std::string::npos)
    {
        auto tail_info = code.substr(start, string::npos);
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
            // no need to re-tune this kernel
            auto compelete_flag = code.find("Antares Tuning Completed in ");
            status->status = (compelete_flag == string::npos) ? "tuning" : "completed";
        }
    }
}

void KernelTuning::submit_tuning_batch_asyc(
    std::vector<std::shared_ptr<GNode>>& nodes,
    std::vector<std::shared_ptr<TuningStatus>>& tuned_kernels,
    std::vector<std::shared_ptr<TuningStatus>>& tuning_kernels)
{
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
        bool symbolic = (FLAGS_fsymbolic && (*gnode)["symbolic"].is_valid_as<bool>());
        NNFUSION_LOG(INFO) << gnode->get_op_type() << " " << gnode->get_name() << ", ir: " << ir;
        if (!ir.empty())
        {
            auto status = std::make_shared<TuningStatus>(gnode);
            status->ir = ir;
            auto response = send_tuning_request(ir, 0, symbolic);
            extract_tunning_status_from_kernel(response, status);

            if (status->status == "" || status->status.empty())
            {
                // submit a new tuning task
                NNFUSION_LOG(INFO) << gnode->get_op_type() << ", ir: " << ir;
                auto response = send_tuning_request(ir, FLAGS_fkernel_tuning_steps, symbolic);
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
}

void KernelTuning::tuning_kernels_sync(std::vector<std::shared_ptr<GNode>>& nodes,
                                       std::vector<std::shared_ptr<TuningStatus>>& tuned_kernels)
{
    size_t num_kernels = nodes.size();
    size_t id = 0;
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
        //NNFUSION_LOG(INFO) << gnode->get_op_type() << ", ir: " << ir;
        if (!ir.empty())
        {
            auto s = std::make_shared<TuningStatus>(gnode);
            s->ir = ir;

            std::string cache_folder = "./kernel_cache";
            struct stat stats;
            if (stat(cache_folder.c_str(), &stats) != 0)
            {
                std::string cmd_create_folder = "mkdir -p " + cache_folder;
                int sys_ret = system(cmd_create_folder.c_str());
            }
            std::string antares_backend =
                get_antares_device_type(n_device_type, FLAGS_ftuning_platform);

            std::string file_id = sha256(ir);
            auto file_name = cache_folder + "/" + file_id + "." + antares_backend + ".c";
            bool symbolic = (FLAGS_fsymbolic && (*gnode)["symbolic"].is_valid_as<bool>());

            std::string cmd = "PROGRESS=1 BACKEND=";
            cmd += antares_backend;
            if (symbolic)
                cmd += " TVM=0";
            if (FLAGS_ftuning_agent.size() > 0)
                cmd += (" AGENT_URL=" + FLAGS_ftuning_agent);
            cmd += " COMPUTE_V1='";
            cmd += ir;
            cmd += ("' antares save " + file_name);

            if (stat(file_name.c_str(), &stats) != 0)
            {
                // generate default kernel
                int sys_ret = system(("STEP=0 " + cmd).c_str());
            }

            std::ifstream ifs(file_name);
            std::string code((std::istreambuf_iterator<char>(ifs)),
                             (std::istreambuf_iterator<char>()));
            extract_tunning_status_from_kernel(code, s);

            if (s->status == "completed")
            {
                std::cout << "\nTuning [" << id++ << "/" << num_kernels
                          << " ops]: op=" << s->op_type << ", name="
                          << ((s->op_name.size() > 26) ? (s->op_name.substr(0, 24) + "..")
                                                       : s->op_name)
                          << ": USE CACHE KERNEL." << std::endl;
                tuned_kernels.push_back(s);
                continue;
            }

            std::cout << "\nTuning [" << id++ << "/" << num_kernels << " ops]: op=" << s->op_type
                      << ", name="
                      << ((s->op_name.size() > 26) ? (s->op_name.substr(0, 24) + "..") : s->op_name)
                      << ":" << std::endl;

            // tuning kernel
            cmd = ("COMMIT=force STEP=" + std::to_string(FLAGS_fkernel_tuning_steps) + " " + cmd);
            int sys_ret = system(cmd.c_str());
            {
                std::ifstream ifs(file_name);
                std::string code((std::istreambuf_iterator<char>(ifs)),
                                 (std::istreambuf_iterator<char>()));
                extract_tunning_status_from_kernel(code, s);
                tuned_kernels.push_back(s);
            }
        }
    }
}

void load_irs_and_tune_kernels_sync(std::string filename,
                                    std::vector<std::shared_ptr<TuningStatus>>& tuned_kernels)
{
    std::ifstream fin(filename);
    std::vector<std::string> tuning_irs;
    if (fin.is_open())
    {
        std::string line;
        while (std::getline(fin, line))
        {
            tuning_irs.push_back(line);
        }
        fin.close();
    }

    size_t id = 0;
    size_t num_kernels = tuning_irs.size();
    for (auto line : tuning_irs)
    {
        auto items = split_string(line, "|");
        NNFUSION_CHECK(items.size() == 5);
        auto op = items[0];
        auto name = items[1];
        bool symbolic = atoi(items[2].c_str());
        size_t cnt = atoi(items[3].c_str());
        auto ir = items[4];

        auto s = std::make_shared<TuningStatus>(op, name, symbolic);
        s->ir = ir;

        std::string cache_folder = "./kernel_cache";
        struct stat stats;
        if (stat(cache_folder.c_str(), &stats) != 0)
        {
            std::string cmd_create_folder = "mkdir -p " + cache_folder;
            int sys_ret = system(cmd_create_folder.c_str());
        }

        std::string file_id = sha256(ir);
        auto antares_backend =
            get_antares_device_type(get_device_type(FLAGS_fdefault_device), FLAGS_ftuning_platform);
        auto file_name = cache_folder + "/" + file_id + "." + antares_backend + ".c";

        std::string cmd = "PROGRESS=1 BACKEND=";
        cmd += antares_backend;
        if (FLAGS_ftuning_agent.size() > 0)
            cmd += (" AGENT_URL=" + FLAGS_ftuning_agent);
        if (symbolic)
            cmd += " TVM=0";
        cmd += " COMPUTE_V1='";
        cmd += ir;
        cmd += ("' antares save " + file_name);

        if (stat(file_name.c_str(), &stats) != 0)
        {
            // generate default kernel
            int sys_ret = system(("STEP=0 " + cmd).c_str());
        }

        // qurey cached kernel

        std::ifstream ifs(file_name);
        if (ifs.is_open())
        {
            std::string code((std::istreambuf_iterator<char>(ifs)),
                             (std::istreambuf_iterator<char>()));
            extract_tunning_status_from_kernel(code, s);
            ifs.close();
        }

        if (FLAGS_fretuning_bar > 0)
        {
            if (s->best_perf <= FLAGS_fretuning_bar)
            {
                std::cout << "\nTuning [" << id++ << "/" << num_kernels
                          << " ops]: op=" << s->op_type << ", name="
                          << ((s->op_name.size() > 26) ? (s->op_name.substr(0, 24) + "..")
                                                       : s->op_name)
                          << ": Match Retuning Bar: " << s->best_perf << "|" << FLAGS_fretuning_bar
                          << std::endl;
                tuned_kernels.push_back(s);
                continue;
            }
        }
        if (s->status == "completed")
        {
            std::cout << "\nTuning [" << id++ << "/" << num_kernels << " ops]: op=" << s->op_type
                      << ", name="
                      << ((s->op_name.size() > 26) ? (s->op_name.substr(0, 24) + "..") : s->op_name)
                      << ": USE CACHE KERNEL." << std::endl;
            tuned_kernels.push_back(s);
            continue;
        }

        std::cout << "\nTuning [" << id++ << "/" << num_kernels << " ops]: op=" << s->op_type
                  << ", name="
                  << ((s->op_name.size() > 26) ? (s->op_name.substr(0, 24) + "..") : s->op_name)
                  << ":" << std::endl;

        // tuning kernel
        cmd = ("COMMIT=force STEP=" + std::to_string(FLAGS_fkernel_tuning_steps) + " " + cmd);
        NNFUSION_LOG(INFO) << cmd;
        auto sys_ret = system(cmd.c_str());
        {
            std::ifstream ifs(file_name);
            std::string code((std::istreambuf_iterator<char>(ifs)),
                             (std::istreambuf_iterator<char>()));
            extract_tunning_status_from_kernel(code, s);
            tuned_kernels.push_back(s);
        }
    }
}

bool KernelTuning::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (FLAGS_fantares_mode)
    {
        parse_allow_and_block_list();
        // register antares kernels anyway here in case kernel selection pass will use them
        register_antares_kernel();
    }

    if (FLAGS_fkernel_tuning_steps <= 0 || !FLAGS_fantares_mode)
    {
        return true;
    }

    std::vector<std::shared_ptr<TuningStatus>> tuned_kernels;
    std::vector<std::shared_ptr<TuningStatus>> tuning_kernels;
    std::unordered_map<std::string, size_t> ir2cnt;
    std::vector<std::shared_ptr<GNode>> nodes;
    std::tie(nodes, tuned_kernels) =
        get_tuning_candidates(graph, m_allow_list, m_block_list, ir2cnt);

    std::string param_str;
    auto dim_infos = graph->get_dim_params();
    for (auto pair : dim_infos)
    {
        param_str += ("_" + pair.first + pair.second.debug_string());
    }

    const std::string dump_file = "./antares_irs" + param_str + ".txt";
    if (FLAGS_fdump_and_tune_irs == 1)
    {
        dump_tuning_irs(dump_file, nodes, ir2cnt);
        //exit(0);
    }

    if (FLAGS_fdump_and_tune_irs == 2)
    {
        load_irs_and_tune_kernels_sync(dump_file, tuned_kernels);
        exit(0);
    }

    if (FLAGS_fantares_codegen_server.size() > 0)
    {
        // Note: currently we asume IP:PORT as the static server and IP:PORT+1 as the symbolic server
        m_static_tuning_server = FLAGS_fantares_codegen_server;

        auto items = split_string(m_static_tuning_server, ":");
        NNFUSION_CHECK(items.size() == 2) << "Wrong server format: " << m_static_tuning_server;
        auto port = atoi(items[1].c_str());
        m_dynamic_tuning_server = items[0] + ":" + std::to_string(port + 1);

        submit_tuning_batch_asyc(nodes, tuned_kernels, tuning_kernels);
    }
    else
    {
        tuning_kernels_sync(nodes, tuned_kernels);
    }
    dump_perf(FLAGS_fantares_perf_file, tuned_kernels, ir2cnt);
    if (FLAGS_fdefault_device == "CUDA" && !FLAGS_fsymbolic)
    {
        insert_to_kernel_cache(nodes);
    }
    return true;
}

void KernelTuning::register_single_kernel(const std::string& op_name)
{
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

bool KernelTuning::register_antares_kernel()
{
    for (auto pair : nnfusion::op::get_op_configs())
    {
        std::string op_name = pair.first;
        std::vector<NNFusion_DeviceType> devs{CUDA_GPU, GENERIC_CPU, HLSL};

        // skip op not in allow_list
        if (m_allow_list.size() > 0 && m_allow_list.find(op_name) == m_allow_list.end())
        {
            continue;
        }
        // skip op in BlockList
        if (m_allow_list.size() == 0 && m_block_list.find(op_name) != m_block_list.end())
        {
            continue;
        }
        register_single_kernel(op_name);
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
