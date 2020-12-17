// Microsoft (c) 2019, NNFusion Team

#include "pattern_substitution.hpp"
#include <queue>
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"

using namespace nnfusion;
using namespace nnfusion::pass::graph;
using namespace nnfusion::profiler;

DEFINE_bool(fpattern_substitution,
            true,
            "Substitute listed patterns with more efficient implementations");

// Only serial pattern supported in current implementation
// The substitution directly applied to computation graph, no back propagation involved
const static std::vector<std::vector<std::string>> PATTERNS = {
    {"Convolution", "BatchNormInference", "Relu"},
    {"Convolution", "BatchNormInference", "Add"},
    {"Convolution", "BatchNormInference"},
    // {"Convolution", "Add", "Reshape", "Relu"},
    {"Convolution", "Add", "Relu"},
    {"Convolution", "Relu"}};

REGISTER_OP(Matched_Pattern)
    // .attr<nnfusion::op::OpConfig::any>("out_shape")
    .infershape([](std::shared_ptr<GNode> gnode) -> void {
        // auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        // Shape out_shape = op->localOpConfig.getRoot()["out_shape"];
        // gnode->set_output_type_and_shape(0, element::f32, out_shape);
    });

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
}

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
            identifier += generate_identifier(ctx);
        }
        if (identifier != "")
        {
            // Todo: more tags, more platform
            std::set<std::string> tags = {"fast"};
            auto fetched_kernel = kernel_db->fetch_with_tags(identifier, "CUDA", tags);
            if (fetched_kernel.function != "")
            {
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
        for (const auto& in_edge : matched.front()->node->get_in_edges())
        {
            auto input_id = in_edge->is_control_edge() ? Graph::kControlSlot : next_input_id++;
            if (input_id != Graph::kControlSlot)
            {
                subs_node->set_input(
                    input_id, matched.front()->node->get_inputs().at(in_edge->get_dst_input()));
            }
            m_graph->add_edge(in_edge->get_src(), in_edge->get_src_output(), subs_node, input_id);
        }

        for (auto m_node : matched)
        {
            // bias as extra inputs
            // here we apply the BN folding to remove computation
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