// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/op.hpp"
DECLARE_bool(fsymbolic);
namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            struct TuningStatus
            {
                TuningStatus(std::shared_ptr<nnfusion::graph::GNode> gnode)
                    : op_type(gnode->get_op_type())
                    , op_name(gnode->get_op_ptr()->get_name())
                    , progress_step(0)
                    , best_perf(-1.0)
                {
                    if (FLAGS_fsymbolic && (*gnode)["symbolic"].is_valid_as<bool>())
                    {
                        op_type = op_type + "*";
                    }
                }
                TuningStatus(std::string op, std::string name, bool symbolic = false)
                    : op_type(op)
                    , op_name(name)
                    , progress_step(0)
                    , best_perf(-1.0)
                {
                    if (symbolic)
                    {
                        op_type = op_type + "*";
                    }
                }
                std::string op_type;
                std::string op_name;
                std::string status;
                int64_t progress_step;
                double best_perf;
                std::string ir;
            };

            class KernelTuning : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;
                static void register_single_kernel(const std::string& op_name);
                bool register_antares_kernel();

            private:
                bool parse_allow_and_block_list();
                void submit_tuning_batch_asyc(
                    std::vector<std::shared_ptr<nnfusion::graph::GNode>>& nodes,
                    std::vector<std::shared_ptr<TuningStatus>>& tuned_kernels,
                    std::vector<std::shared_ptr<TuningStatus>>& tuning_kernels);
                void
                    tuning_kernels_sync(std::vector<std::shared_ptr<nnfusion::graph::GNode>>& nodes,
                                        std::vector<std::shared_ptr<TuningStatus>>& tuned_kernels);
                bool insert_to_kernel_cache(
                    const std::vector<std::shared_ptr<nnfusion::graph::GNode>>& nodes);
                std::string
                    send_tuning_request(std::string& ir, int64_t step, bool symbolic = false);

            private:
                std::unordered_set<std::string> m_block_list;
                std::unordered_set<std::string> m_allow_list;
                std::string m_static_tuning_server;
                std::string m_dynamic_tuning_server;
            };
        }
    }
} // namespace nnfusion