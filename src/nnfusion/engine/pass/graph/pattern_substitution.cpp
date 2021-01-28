// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pattern_substitution.hpp"
#include <queue>
#include "nnfusion/common/common.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"
#include "nnfusion/core/operators/op_define/fused.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"
#include "nnfusion/engine/cache/manager.hpp"
#include "nnfusion/engine/op.hpp"
#include "nnfusion/engine/pass/graph/kernel_selection.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion;
using namespace nnfusion::pass::graph;
using namespace nnfusion::profiler;

DEFINE_bool(fpattern_substitution,
            true,
            "Substitute listed patterns with more efficient implementations");
DEFINE_bool(fbiasadd_fix,
            false,
            "Fix biasadd shape for TVM Conv2d-Add fusion in pattern_substitution_pass");

// Only serial pattern supported in current implementation
// The substitution directly applied to computation graph, no back propagation involved
const static std::vector<std::vector<std::string>> PATTERNS = {
    // {"Convolution", "BatchNormInference", "Relu"},
    // {"Convolution", "BatchNormInference", "Add"},
    // {"Convolution", "BatchNormInference"},
    // Conv-BN-Relu is converted into Conv-Add-Relu
    {"Convolution", "Add", "Relu"},
    {"Convolution", "Relu"}};

REGISTER_OP(Matched_Pattern)
    // .attr<nnfusion::op::OpConfig::any>("out_shape")
    .infershape([](std::shared_ptr<GNode> gnode) -> void {
        // auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        // Shape out_shape = op->localOpConfig.getRoot()["out_shape"];
        // gnode->set_output_type_and_shape(0, element::f32, out_shape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Fused>(curr->get_op_ptr());
        return _op->get_fused_ir2() + _op->get_plan_rule();
    });
;

namespace
{
    struct TaggedNode
    {
        TaggedNode()
            : node(nullptr)
            , ready_inputs(0)
            , visited(false)
        {
        }

        std::shared_ptr<GNode> node;
        size_t ready_inputs;
        bool visited;
    };
} // namespace

class PatternOptimizer
{
public:
    PatternOptimizer(std::shared_ptr<Graph> g,
                     std::vector<std::string> p,
                     shared_ptr<cache::KernelCacheManager> db)
        : m_graph(g)
        , m_pattern(p)
        , kernel_db(db)
    {
        m_nodes.resize(m_graph->get_max_node_id());
    }

    void MatchAndSubstitute()
    {
        std::queue<size_t> ready;
        for (auto node : m_graph->get_ordered_ops())
        {
            size_t id = node->get_id();
            m_nodes[id] = std::make_shared<TaggedNode>();
            m_nodes[id]->node = node;
            if (!(m_nodes[id]->visited) &&
                (m_nodes[id]->ready_inputs == node->get_in_edges().size()))
            {
                ready.push(id);
            }
        }

        while (!ready.empty())
        {
            size_t node_id = ready.front();
            ready.pop();

            node_id = MatchPattern(node_id);
            auto tn = m_nodes[node_id];
            tn->visited = true;
            for (auto edge : tn->node->get_out_edges())
            {
                auto dst = m_nodes[edge->get_dst()->get_id()];
                // node that will not be computed
                if (!dst)
                    continue;
                dst->ready_inputs++;
                NNFUSION_CHECK(!(dst->visited));
                if (dst->ready_inputs >= dst->node->get_in_edges().size())
                {
                    NNFUSION_CHECK(dst->ready_inputs == dst->node->get_in_edges().size());
                    ready.push(dst->node->get_id());
                }
            }
        }
    }

private:
    size_t MatchPattern(size_t id)
    {
        std::shared_ptr<TaggedNode> tn = m_nodes[id];
        std::vector<std::shared_ptr<TaggedNode>> matched;

        for (size_t i = 0; i < m_pattern.size(); i++)
        {
            if (tn && tn->node->get_op_type() == m_pattern[i])
            {
                if (i == m_pattern.size() - 1 || tn->node->get_out_edges().size() == 1)
                {
                    matched.push_back(tn);
                    for (auto edge : tn->node->get_out_edges())
                    {
                        tn = m_nodes[edge->get_dst()->get_id()];
                    }
                    continue;
                }
            }
            return id;
        }

        std::string identifier = "";
        for (auto m_tn : matched)
        {
            std::shared_ptr<KernelContext> ctx(new KernelContext(m_tn->node));
            identifier += ctx->generate_identifier();
        }
        if (identifier != "")
        {
            // Todo: more tags, more platform
            std::set<std::string> tags = {};
            auto fetched_kernel = kernel_db->fetch_with_tags(identifier, "CUDA", tags);
            if (fetched_kernel != nullptr)
            {
                NNFUSION_CHECK(fetched_kernel->function != "");
                NNFUSION_LOG(INFO) << "Substitution applied: " << identifier;
                return Substitution(matched, identifier);
            }
        }
        return id;
    }

    size_t Substitution(const std::vector<std::shared_ptr<TaggedNode>>& matched,
                        const std::string& identifier)
    {
        nnfusion::op::OpConfig::any op_config;
        // op_config["out_shape"] = Shape(matched.back()->node->get_output_shape(0));
        auto subs_op = std::make_shared<nnfusion::op::GenericOp>(
            "Matched_Pattern", "Matched_Pattern", op_config);
        GNodeVector empty_inputs;
        auto subs_node = std::make_shared<GNode>(subs_op, empty_inputs);

        m_graph->add_node(subs_node);
        int next_input_id = 0;
        int next_output_id = 0;

        auto front_node = matched.front()->node;
        for (size_t i = 0; i < front_node->get_input_size(); i++)
        {
            const auto in_edge = front_node->get_in_edge(i);
            subs_node->set_input(next_input_id, front_node->get_inputs().at(i));
            m_graph->add_edge(
                in_edge->get_src(), in_edge->get_src_output(), subs_node, next_input_id);
            next_input_id++;
        }

        for (const auto& in_edge : front_node->get_in_edges())
            if (in_edge->is_control_edge())
                m_graph->add_edge(
                    in_edge->get_src(), in_edge->get_src_output(), subs_node, Graph::kControlSlot);

        for (auto m_node : matched)
        {
            // bias as extra inputs
            // here we apply the BN folding to remove computation

            // biasadd fix for TVM conv-biasadd fusion due to different implementation of BiasAdd in NNFusion and TVM
            if (m_node->node->get_op_type() == "Add" && FLAGS_fbiasadd_fix)
            {
                if (m_pattern[0] == "Convolution" && m_pattern[1] == "Add")
                {
                    auto add_node = m_node->node;
                    auto add_bias_node = add_node->get_in_edge(1)->get_src();
                    auto add_bias_const_ptr = std::dynamic_pointer_cast<nnfusion::op::Constant>(
                        add_bias_node->get_op_ptr());
                    auto dtype = add_node->get_input_element_type(1);
                    std::vector<double> add_bias = ExtractConstantData(add_bias_const_ptr, dtype);
                    auto bias_shape = add_node->get_input_shape(1);
                    std::shared_ptr<op::Constant> new_add_bias_op_ptr;
                    if (dtype == element::f32)
                    {
                        auto add_bias_converted = ConvertAddBias<float>(add_bias, bias_shape);
                        new_add_bias_op_ptr = std::make_shared<op::Constant>(
                            dtype, add_node->get_input_shape(1), add_bias_converted.data());
                    }
                    else
                    {
                        NNFUSION_CHECK_FAIL() << "Not support DataType " << dtype;
                    }
                    auto new_add_bias_gnode = std::make_shared<nnfusion::graph::GNode>(
                        new_add_bias_op_ptr, GNodeVector());
                    m_graph->replace_node(add_bias_node, new_add_bias_gnode, false);
                }
            }
            if (m_node->node->get_op_type() == "BatchNormInference" ||
                m_node->node->get_op_type() == "Add")
            {
                auto bias_edge = m_node->node->get_in_edge(1);
                subs_node->set_input(next_input_id, m_node->node->get_inputs().at(1));
                m_graph->add_edge(
                    bias_edge->get_src(), bias_edge->get_src_output(), subs_node, next_input_id++);
            }
        }

        // set the I/O information for new node
        subs_node->set_output_type_and_shape(
            0, subs_node->get_input_element_type(0), matched.back()->node->get_output_shape(0));

        // dedup the output in advance
        std::unordered_map<std::string, size_t> node_outputs;
        for (const auto& out_edge : matched.back()->node->get_out_edges())
        {
            auto output_id = out_edge->is_control_edge() ? Graph::kControlSlot : next_output_id;
            if (output_id != Graph::kControlSlot)
            {
                auto output_name = matched.back()
                                       ->node->get_outputs()
                                       .at(out_edge->get_src_output())
                                       ->get_tensor_ptr()
                                       ->get_name();
                auto iter = node_outputs.find(output_name);
                if (iter == node_outputs.end())
                {
                    subs_node->set_output(
                        output_id,
                        matched.back()->node->get_outputs().at(out_edge->get_src_output()));
                    node_outputs[output_name] = next_output_id++;
                }
                else
                {
                    output_id = node_outputs[output_name];
                }
            }
            m_graph->add_edge(subs_node, output_id, out_edge->get_dst(), out_edge->get_dst_input());
        }
        for (auto tn : matched)
        {
            m_graph->remove_node(tn->node);
        }

        (*subs_node)["identifier"] = identifier;

        m_nodes.resize(m_graph->get_max_node_id());
        size_t id = subs_node->get_id();
        m_nodes[id] = std::make_shared<TaggedNode>();
        m_nodes[id]->node = subs_node;
        return id;
    }

    std::vector<double> ExtractConstantData(const std::shared_ptr<op::Constant> ptr,
                                            const element::Type& dtype)
    {
        if (dtype == element::f64)
        {
            auto data_vec = ptr->get_vector<double>();
            std::vector<double> ret;
            for (auto i = 0; i < data_vec.size(); i++)
            {
                ret.push_back((double)data_vec[i]);
            }
            return ret;
        }
        else if (dtype == element::f32)
        {
            auto data_vec = ptr->get_vector<float>();
            std::vector<double> ret;
            for (auto i = 0; i < data_vec.size(); i++)
            {
                ret.push_back((double)data_vec[i]);
            }
            return ret;
        }
        else if (dtype == element::bf16)
        {
            auto data_vec = bfloat16::to_float_vector(ptr->get_vector<bfloat16>());
            std::vector<double> ret;
            for (auto i = 0; i < data_vec.size(); i++)
            {
                ret.push_back((double)data_vec[i]);
            }
            return ret;
        }
        else if (dtype == element::i64)
        {
            auto data_vec = ptr->get_vector<float>();
            std::vector<double> ret;
            for (auto i = 0; i < data_vec.size(); i++)
            {
                ret.push_back((double)data_vec[i]);
            }
            return ret;
        }
        else if (dtype == element::i32)
        {
            auto data_vec = ptr->get_vector<float>();
            std::vector<double> ret;
            for (auto i = 0; i < data_vec.size(); i++)
            {
                ret.push_back((double)data_vec[i]);
            }
            return ret;
        }
        else if (dtype == element::i16)
        {
            auto data_vec = ptr->get_vector<float>();
            std::vector<double> ret;
            for (auto i = 0; i < data_vec.size(); i++)
            {
                ret.push_back((double)data_vec[i]);
            }
            return ret;
        }
        else if (dtype == element::i8)
        {
            auto data_vec = ptr->get_vector<float>();
            std::vector<double> ret;
            for (auto i = 0; i < data_vec.size(); i++)
            {
                ret.push_back((double)data_vec[i]);
            }
            return ret;
        }

        NNFUSION_CHECK_FAIL() << "Not support DataType " << dtype;
        std::vector<double> ret;
        return ret;
    }

    template <typename T>
    std::vector<T> ConvertAddBias(const std::vector<double>& bias, const nnfusion::Shape bias_shape)
    {
        std::vector<T> bias_output;
        bias_output.resize(bias.size());

        auto n = bias_shape[0];
        auto c = bias_shape[1];
        auto h = bias_shape[2];
        auto w = bias_shape[3];

        std::vector<double> bias_extracted;
        bias_extracted.resize(c);
        for (auto i = 0; i < c; i++)
        {
            bias_extracted[i] = bias[i * h * w];
        }

        for (auto i = 0; i < bias.size(); i++)
        {
            bias_output[i] = (T)bias_extracted[i % c];
        }

        return bias_output;
    }

    std::shared_ptr<Graph> m_graph;
    std::vector<std::string> m_pattern;
    std::vector<std::shared_ptr<TaggedNode>> m_nodes;
    std::shared_ptr<cache::KernelCacheManager> kernel_db;
};

bool PatternSubstitutionPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    auto kernel_db = std::make_shared<cache::KernelCacheManager>();
    if (!kernel_db->is_valid())
    {
        NNFUSION_LOG(INFO) << "No valid kernel cache, no pattern substitution will be processed";
        return true;
    }
    bool substitution = FLAGS_fpattern_substitution;
    if (substitution)
    {
        for (auto pattern : PATTERNS)
        {
            PatternOptimizer optimizer(graph, pattern, kernel_db);
            optimizer.MatchAndSubstitute();
        }
    }
    return true;
}