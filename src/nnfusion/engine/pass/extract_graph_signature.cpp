// Microsoft (c) 2019, Yanhui Hong
#include "extract_graph_signature.hpp"

using namespace nnfusion::interpreter;

bool ExtractGraphSignature::extract_result(std::shared_ptr<TranslationUnit> tu,
                                           std::shared_ptr<graph::Graph> graph)
{
    for (auto gnode : graph->get_outputs())
    {
        std::shared_ptr<nnfusion::descriptor::Tensor> tv = gnode->get_output_tensor_ptr(0);
        CHECK_NOT_NULLPTR(tv);
        tu->output_names->insert(tv->get_name());
        LOG(INFO) << "Result Tensor: " << tv->get_name();
    }
    return true;
}

bool ExtractGraphSignature::extract_constants(std::shared_ptr<InterpreterContext> ctx,
                                              std::shared_ptr<TranslationUnit> tu,
                                              std::shared_ptr<graph::Graph> graph)
{
    for (auto gnode : graph->get_nodes())
    {
        if (dynamic_cast<nnfusion::op::Constant*>(gnode->get_op_ptr().get()))
        {
            shared_ptr<nnfusion::descriptor::Tensor> tv = gnode->get_output_tensor_ptr(0);
            CHECK_NOT_NULLPTR(tv);
            tu->constants->insert(tv);

            LOG(INFO) << "Constant Tensor: " << tv->get_name();
        }
    }
    return true;
}

void ExtractGraphSignature::propagate_in_place_input(std::shared_ptr<InterpreterContext> ctx,
                                                     NodeOut nodeOutput,
                                                     std::string input_name)
{
    std::deque<NodeOut> stack;
    stack.push_front(nodeOutput);

    while (stack.size() > 0)
    {
        auto it = stack.front();
        stack.pop_front();
        for (auto edge : it.node->get_output_users(it.index))
        {
            auto out_node = edge->get_dst();
            auto c_op = std::dynamic_pointer_cast<nnfusion::op::Op>(out_node->get_op_ptr());
            if (!c_op || c_op->is_output())
            {
                continue;
            }

            if (auto op_annotations = c_op->get_op_annotations())
            {
                for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
                {
                    if (oi_pair.input == edge->get_src_output() && !oi_pair.destructive)
                    {
                        size_t output_index = oi_pair.output;
                        auto& output_tensor = out_node->get_output_tensor(output_index);

                        ctx->m_variable_name_map[output_tensor.get_name()] = input_name;

                        LOG(INFO) << "GPU codegen: Forwarding " << input_name << " through "
                                  << output_tensor.get_name();
                        stack.push_back(NodeOut(out_node, output_index));
                    }
                }
            }
        }
    }
}

void ExtractGraphSignature::propagate_in_place_output(std::shared_ptr<InterpreterContext> ctx,
                                                      NodeOut nodeOutput,
                                                      std::string output_name)
{
    // we start with a particular output
    // which is an argument to a given op::Result
    size_t offset = nodeOutput.node->get_output_tensor(nodeOutput.index).get_pool_offset();
    auto it = nodeOutput;

    bool propagate_further = false;
    do
    {
        propagate_further = false;
        auto arg = std::dynamic_pointer_cast<nnfusion::op::Op>(it.node->get_op_ptr());
        if (!arg)
        {
            break;
        }
        if (auto op_annotations = arg->get_op_annotations())
        {
            for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
            {
                if (oi_pair.output == it.index)
                {
                    size_t input_index = oi_pair.input;
                    auto in_edge = it.node->get_in_edge(input_index);
                    CHECK_NOT_NULLPTR(in_edge);
                    auto tmp_node = in_edge->get_src();
                    auto& input_tensor = tmp_node->get_output_tensor(in_edge->get_src_output());
                    if (input_tensor.get_pool_offset() == offset &&
                        !tmp_node->get_op_ptr()->is_parameter() &&
                        !tmp_node->get_op_ptr()->is_constant())
                    {
                        LOG(INFO) << "Reusing " << output_name << " for "
                                  << input_tensor.get_name();

                        ctx->m_variable_name_map[input_tensor.get_name()] = output_name;

                        it = NodeOut(tmp_node, in_edge->get_src_output());
                        propagate_further = true;
                    }
                }
            }
        }
    } while (propagate_further);
}

bool ExtractGraphSignature::extract_args(std::shared_ptr<InterpreterContext> ctx,
                                         std::shared_ptr<TranslationUnit> tu,
                                         std::shared_ptr<graph::Graph> graph)
{
    size_t arg_index = 0;
    for (auto gnode : graph->get_parameters())
    {
        for (size_t i = 0; i < gnode->get_output_size(); ++i)
        {
            auto tv = gnode->get_output_tensor_ptr(i);
            CHECK_NOT_NULLPTR(tv);
            tu->arg.push_back(tv);
            const element::Type& et = tv->get_element_type();

            string type = et.c_type_string();
            stringstream ss;
            ss << "((" << type << "*)(inputs[" << arg_index << "]))";
            ctx->m_variable_name_map[tv->get_name()] = ss.str();
            propagate_in_place_input(ctx, NodeOut(gnode, i), ss.str());

            arg_index++;

            LOG(INFO) << "Param Tensor:\t" << tv->get_name() << "\twith id: " << ss.str();
        }
    }
    return true;
}

bool ExtractGraphSignature::extract_output(std::shared_ptr<InterpreterContext> ctx,
                                           std::shared_ptr<TranslationUnit> tu,
                                           std::shared_ptr<graph::Graph> graph)
{
    for (size_t i = 0; i < graph->get_output_size(); ++i)
    {
        auto node = graph->get_output_op(i);

        auto res = dynamic_pointer_cast<nnfusion::op::Result>(node->get_op_ptr());
        if (!res->needs_copy_to_host())
        {
            continue;
        }

        shared_ptr<nnfusion::descriptor::Tensor> tv = node->get_output_tensor_ptr(0);
        CHECK_NOT_NULLPTR(tv);

        tu->out.push_back(tv);

        string type = tv->get_element_type().c_type_string();
        stringstream ss;
        ss << "((" << type << "*)(outputs[" << i << "]))";
        ctx->m_variable_name_map[tv->get_name()] = ss.str();
        //keep assigning different outputs to a result descriptor
        //op::Result emitter will check if in and out descriptors are the same
        //and skip a copy
        auto in_edge = node->get_in_edge(0);
        CHECK_NOT_NULLPTR(in_edge);
        auto input_node = in_edge->get_src();
        if (!input_node->get_op_ptr()->is_constant() && !input_node->get_op_ptr()->is_parameter())
        {
            shared_ptr<nnfusion::descriptor::Tensor> itv =
                input_node->get_output_tensor_ptr(in_edge->get_src_output());
            auto output_name = ss.str();
            ctx->m_variable_name_map[itv->get_name()] = output_name;
            propagate_in_place_output(
                ctx, NodeOut(input_node, in_edge->get_src_output()), output_name);
            LOG(INFO) << "Output Tensor:\t" << itv->get_name() << "\t with id:" << output_name;
        }
    }
    return true;
}

bool ExtractGraphSignature::run(std::shared_ptr<InterpreterContext> ctx,
                                std::shared_ptr<TranslationUnit> tu)
{
    auto graph = tu->m_graph;
    tu->memory_pool_size = graph->get_temporary_pool_size();
    CHECK(extract_result(tu, graph));
    CHECK(extract_constants(ctx, tu, graph));
    CHECK(extract_args(ctx, tu, graph));
    CHECK(extract_output(ctx, tu, graph));
    return true;
}
