// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"
#include "nnfusion/util/curl_request.hpp"

using namespace nnfusion::graph;

DECLARE_string(fdefault_device);
DECLARE_string(fantares_codegen_server);

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class GraphCoreCodegenPass : public GraphPassBase
            {
                std::string currentBackend;
                std::string autogen(const std::string& expr)
                {
                    if (FLAGS_fantares_codegen_server == "")
                        FLAGS_fantares_codegen_server = "10.150.145.98:8883";
                    static std::unordered_map<std::string, std::string> code_cache;
                    std::string response;
                    auto it = code_cache.find(expr);
                    if (it == code_cache.end())
                    {
                        CurlRequest req(FLAGS_fantares_codegen_server);
                        req.add_custom_header(("COMPUTE_V1: " + expr).c_str());
                        req.add_custom_header("ARGS: ");

                        printf("[Autogen] %s\n", expr.c_str());
                        NNFUSION_CHECK(true == req.send_request(response));
                        NNFUSION_CHECK(strncmp(response.c_str(), "[ERROR]", 7) != 0) << expr << "\n"
                                                                                     << response;
                        code_cache[expr] = response;
                        return std::move(response);
                    }
                    else
                        return it->second;
                }

                template <class T1, class T2>
                inline std::string
                    join_collections(const T1& vect, T2 func, bool skip_empty = false)
                {
                    std::stringstream result;
                    int idx = 0;
                    for (auto& it : vect)
                    {
                        auto str = func(idx, it);
                        if (!str.size() && skip_empty)
                            continue;
                        if (idx > 0)
                            result << ", ";
                        result << str;
                        ++idx;
                    }
                    return result.str();
                }

                // inline int get_type_id(nnfusion::element::Type type)
                // {
                //     // TODO: fill more type cases
                //     if (type == nnfusion::element::f32)
                //         return DT_FLOAT;
                //     throw std::runtime_error("Not supported element type.");
                // }

                template <class T>
                inline std::shared_ptr<T> get_op_object(std::shared_ptr<GNode>& curr)
                {
                    auto _op = static_pointer_cast<T>(curr->get_op_ptr());
                    NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not "
                                                    << curr->get_op_ptr()->get_op_type();
                    return _op;
                }

                inline void UNHANDLED_CASE(std::shared_ptr<GNode>& curr)
                {
                    printf("## Unhandled case for %s:\n",
                           curr->get_op_ptr()->get_op_type().c_str());
                    for (int i = 0; i < curr->get_input_size(); ++i)
                        printf(">> in-%d : %s\n",
                               i,
                               vector_to_string(curr->get_input_shape(i)).c_str());
                    for (int i = 0; i < curr->get_output_size(); ++i)
                        printf(">> out-%d: %s\n",
                               i,
                               vector_to_string(curr->get_output_shape(i)).c_str());
                    exit(1);
                };

            public:
                bool run_on_graph(std::shared_ptr<Graph>& graph) override
                {
                    currentBackend = "graphcore";

                    NNFUSION_LOG(INFO) << "Codegen for " << currentBackend << " starts up.";

                    auto nodes = graph->get_nodes();
                    std::unordered_map<std::shared_ptr<GNode>, int> din, dout;

                    // Count degrees
                    for (auto& it : nodes)
                    {
                        for (auto& in_edge : it->get_in_edges())
                        {
                            if (in_edge->is_control_edge())
                                continue;
                            NNFUSION_CHECK(in_edge->get_dst() == it);
                            din[it]++;
                            dout[in_edge->get_src()]++;
                        }
                    }

                    // Name nodes, legality checks
                    std::unordered_set<std::shared_ptr<GNode>> visited, vis_pend, blacklist;
                    std::unordered_set<std::string> name_used;
                    std::unordered_map<std::shared_ptr<GNode>, std::string> arg_names;
                    for (auto& it : nodes)
                    {
                        NNFUSION_CHECK(it.get() != nullptr);

                        auto arg_name = "Z0_" + it->get_op_ptr()->get_op_type() + "_" +
                                        it->get_op_ptr()->get_name();
                        for (auto& c : arg_name)
                            if (!isalpha(c) && !isdigit(c))
                                c = '_';
                        if (name_used.count(arg_name))
                        {
                            for (int i = 1;; ++i)
                            {
                                auto alter = arg_name + "_" + std::to_string(i);
                                if (!name_used.count(alter))
                                {
                                    arg_name = alter;
                                    break;
                                }
                            }
                        }
                        name_used.insert(arg_name);
                        arg_names[it] = arg_name;

                        if (din[it] == 0 && dout[it] == 0)
                            visited.insert(it), blacklist.insert(it);
                        NNFUSION_CHECK(it->get_output_size() == 1);
                    }
                    NNFUSION_LOG(INFO) << "There are " << blacklist.size()
                                       << " standalone GNode(s) found.";
                    name_used.clear();

                    // Fill offsetup nodes
                    std::deque<std::shared_ptr<GNode>> gen_q, pend_q;
                    for (auto& it : nodes)
                    {
                        if (visited.count(it))
                            continue;
                        if (din[it] == 0)
                        {
                            gen_q.push_back(it);
                        }
                    }

                    NNFUSION_CHECK(
                        0 ==
                        system(("mkdir -p nnfusion_rt/" + currentBackend + "_codegen").c_str()));

                    std::ofstream fout("nnfusion_rt/" + currentBackend + "_codegen/nnfusion_rt.h");

                    fout << "#if 1\n";
                    // Perform blockfusion
                    int offset = 0, step = 0;
                    auto new_super_step = [&]() {
                        while (pend_q.size())
                        {
                            gen_q.push_back(pend_q.front());
                            pend_q.pop_front();
                        }
                        if (offset > 0)
                            ++step, offset = 0;
                    };

                    const int max_threads = 1216 * 6;

                    auto print_standard_kernel_code = [&](
                        std::shared_ptr<GNode>& curr,
                        std::ofstream& fout,
                        const std::string& code,
                        std::vector<int> shards = {},
                        std::vector<std::string> convert_input = {}) {
                        if (!shards.size())
                            shards = std::vector<int>(1 + curr->get_input_size(), 1);
                        if (!convert_input.size())
                            convert_input = std::vector<std::string>(curr->get_input_size());

                        int thread_uses = 1, pos = 0, next;
                        while (next = code.find("// [thread_extent] threadIdx_", pos), next >= 0)
                        {
                            int eq = code.find(" = ", next);
                            NNFUSION_CHECK(eq >= 0);
                            thread_uses *= atoi(code.c_str() + eq + 3);
                            pos = eq;
                        }
                        NNFUSION_CHECK(thread_uses == 1);
                        thread_uses *= shards.back();

                        // if no enough thread_uses, then new_super_step()
                        if (offset + thread_uses > max_threads)
                        {
                            new_super_step();
                            NNFUSION_CHECK(offset + thread_uses <= max_threads);
                        }

                        fout << "Tensor " << arg_names[curr] << " = compute_task(g, {";
                        std::vector<int> range(curr->get_input_size());
                        fout << join_collections(
                                    range,
                                    [&](int idx, int val) {
                                        return arg_names[curr->get_in_edge(idx)->get_src()] +
                                               convert_input[idx];
                                    })
                             << "}, R\"(" << code << ")\", ";
                        fout << step << ", " << offset << ", " << offset + thread_uses << ", {"
                             << join_collections(
                                    shards, [](int idx, int val) { return std::to_string(val); })
                             << "}).reshape({" << join_collections(curr->get_output_shape(0),
                                                                   [&](int idx, ssize_t val) {
                                                                       return std::to_string(val);
                                                                   })
                             << "})"
                             << ";\n";
                        offset += thread_uses;
                    };

                    auto codegen_for_elementwise = [&](std::shared_ptr<GNode>& curr,
                                                       std::ofstream& fout,
                                                       const std::string& topi) {
                        std::string expr = " -";
                        for (int i = 0; i < curr->get_input_size(); ++i)
                            expr += " input(\"input" + std::to_string(i) + "\", @common_shape@);";
                        expr += " output(@common_shape@, " + topi + ");";

                        int num_elements = 1, y;
                        for (auto& it : curr->get_input_shape(0))
                            num_elements *= it;
                        for (int i = max_threads; i >= 1; --i)
                            if (num_elements % i == 0)
                            {
                                y = i;
                                break;
                            }

                        auto code = autogen(op::create_code_from_template(
                            expr,
                            {{"common_shape", "[ " + std::to_string(num_elements / y) + " ]"}}));

                        print_standard_kernel_code(
                            curr, fout, code, std::vector<int>(1 + curr->get_input_size(), y), {});
                    };

                    std::unordered_map<std::string,
                                       std::function<void(std::shared_ptr<GNode>&, std::ofstream&)>>
                        kernel_dict;

                    // Elementwise Ops
                    kernel_dict["Add"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        codegen_for_elementwise(
                            curr, fout, "topi=topi.add(args(\"input0\"), args(\"input1\"))");
                    };
                    kernel_dict["Subtract"] = [&](std::shared_ptr<GNode>& curr,
                                                  std::ofstream& fout) {
                        codegen_for_elementwise(
                            curr, fout, "topi=topi.subtract(args(\"input0\"), args(\"input1\"))");
                    };
                    kernel_dict["Multiply"] = [&](std::shared_ptr<GNode>& curr,
                                                  std::ofstream& fout) {
                        codegen_for_elementwise(
                            curr, fout, "topi=topi.multiply(args(\"input0\"), args(\"input1\"))");
                    };
                    kernel_dict["Divide"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        codegen_for_elementwise(
                            curr, fout, "topi=topi.divide(args(\"input0\"), args(\"input1\"))");
                    };
                    kernel_dict["Power"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        codegen_for_elementwise(
                            curr, fout, "topi=topi.power(args(\"input0\"), args(\"input1\"))");
                    };
                    kernel_dict["LessEq"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        codegen_for_elementwise(
                            curr, fout, "topi=topi.less_equal(args(\"input0\"), args(\"input1\"))");
                    };
                    kernel_dict["Equal"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        codegen_for_elementwise(
                            curr, fout, "topi=topi.equal(args(\"input0\"), args(\"input1\"))");
                    };
                    kernel_dict["Exp"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        codegen_for_elementwise(curr, fout, "topi=topi.exp(args(\"input0\"))");
                    };
                    kernel_dict["Negative"] = [&](std::shared_ptr<GNode>& curr,
                                                  std::ofstream& fout) {
                        codegen_for_elementwise(curr, fout, "topi=topi.negative(args(\"input0\"))");
                    };
                    kernel_dict["Tanh"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        codegen_for_elementwise(curr, fout, "topi=topi.tanh(args(\"input0\"))");
                    };
                    kernel_dict["Relu"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        codegen_for_elementwise(curr, fout, "topi=topi.nn.relu(args(\"input0\"))");
                    };
                    kernel_dict["Relu6"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        codegen_for_elementwise(
                            curr, fout, "topi=topi.clip(args(\"input0\"), 0, 6)");
                    };
                    kernel_dict["Sigmoid"] = [&](std::shared_ptr<GNode>& curr,
                                                 std::ofstream& fout) {
                        codegen_for_elementwise(curr, fout, "topi=topi.sigmoid(args(\"input0\"))");
                    };
                    kernel_dict["Log"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        codegen_for_elementwise(curr, fout, "topi=topi.log(args(\"input0\"))");
                    };

                    // Other Ops
                    kernel_dict["Constant"] = [&](std::shared_ptr<GNode>& curr,
                                                  std::ofstream& fout) {
                        auto p_const = std::dynamic_pointer_cast<op::Constant>(curr->get_op_ptr());
                        NNFUSION_CHECK(p_const != nullptr);
                        const void* dptr = p_const->get_data_ptr();
                        size_t size = p_const->get_data_size();

                        std::vector<std::string> types;
                        if (curr->get_output_element_type(0).c_type_string() == "float")
                            types = {"float", "FLOAT"};
                        else if (curr->get_output_element_type(0).c_type_string() == "int")
                            types = {"unsigned int", "UNSIGNED_INT"};
                        else if (curr->get_output_element_type(0).c_type_string() == "int32_t")
                            types = {"int", "INT"};
                        else if (curr->get_output_element_type(0).c_type_string() == "char")
                            types = {"char", "CHAR"};
                        /*
                        else if (curr->get_output_element_type(0).c_type_string() == "bool")
                        {
                            types = {"int", "INT"};
                            size_t cnt = size / sizeof(bool);
                            int32_t* new_dptr = new int32_t[cnt];
                            bool* dptr_int64 = (bool*)dptr;
                            for (size_t i = 0; i < cnt; i++)
                                new_dptr[i] = dptr_int64[i] ? 1 : 0;
                            dptr = new_dptr;
                            size = cnt * sizeof(int32_t);
                        }
                        */
                        else if (curr->get_output_element_type(0).c_type_string() == "int64_t")
                        {
                            types = {"int", "INT"};
                            // Convert INT64 to INT32
                            // GraphCore only support INT32
                            size_t cnt = size / sizeof(int64_t);
                            int32_t* new_dptr = new int32_t[cnt];
                            int64_t* dptr_int64 = (int64_t*)dptr;
                            for (size_t i = 0; i < cnt; i++)
                            {
                                new_dptr[i] = (int32_t)dptr_int64[i];
                                NNFUSION_CHECK(((uint64_t)new_dptr[i]) <= (uint64_t)dptr_int64[i])
                                    << "Value becomes invalid if converted to int32_t from "
                                       "int64_t.";
                            }

                            dptr = new_dptr;
                            size = cnt * sizeof(int32_t);
                        }
                        else
                        {
                            NNFUSION_LOG(ERROR) << "Unsupported Type: "
                                                << curr->get_output_element_type(0).c_type_string();
                            assert(0);
                        }

                        NNFUSION_CHECK(0 == system(("mkdir -p nnfusion_rt/" + currentBackend +
                                                    "_codegen/Constant")
                                                       .c_str()));
                        FILE* fp = fopen(("nnfusion_rt/" + currentBackend + "_codegen/Constant/" +
                                          arg_names[curr])
                                             .c_str(),
                                         "wb");
                        NNFUSION_CHECK(fp != nullptr);
                        NNFUSION_CHECK(size == fwrite(dptr, 1, size, fp));
                        fclose(fp);

                        if (dptr != p_const->get_data_ptr())
                        {
                            auto buf = (const char*)dptr;
                            delete buf;
                        }

                        fout << "Tensor " << arg_names[curr] << " = load_constant<" << types[0]
                             << ">(g, data_ptrs, " << types[1] << ", {"
                             << join_collections(
                                    curr->get_output_shape(0),
                                    [](int idx, ssize_t it) { return std::to_string(it); })
                             << "}, \"" << arg_names[curr] << "\");";
                    };

                    kernel_dict["Parameter"] = [&](std::shared_ptr<GNode>& curr,
                                                   std::ofstream& fout) {
                        std::vector<std::string> types;
                        if (curr->get_output_element_type(0) == nnfusion::element::f32)
                            types = {"float", "FLOAT", "1.0f"};
                        else if (curr->get_output_element_type(0) == nnfusion::element::i32)
                            types = {"unsigned int", "UNSIGNED_INT", "1"};
                        else
                            assert(0);

                        fout << "Tensor " << arg_names[curr] << " = load_constant<" << types[0]
                             << ">(g, data_ptrs, " << types[1] << ", {"
                             << join_collections(
                                    curr->get_output_shape(0),
                                    [](int idx, ssize_t it) { return std::to_string(it); })
                             << "}, \"" << arg_names[curr] << "\", true);";
                    };

                    kernel_dict["Result"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        fout << "Tensor &" << arg_names[curr] << " = "
                             << arg_names[curr->get_in_edge(0)->get_src()] << ";\n";
                    };

                    kernel_dict["Broadcast"] = [&](std::shared_ptr<GNode>& curr,
                                                   std::ofstream& fout) {
                        auto _op = get_op_object<nnfusion::op::Broadcast>(curr);
                        auto axes = _op->get_broadcast_axes();
                        fout << "Tensor " << arg_names[curr] << " = "
                             << arg_names[curr->get_in_edge(0)->get_src()] << ".reshape({"
                             << join_collections(curr->get_output_shape(0),
                                                 [&](int idx, ssize_t val) {
                                                     return axes.count(idx) ? std::string("1")
                                                                            : std::to_string(val);
                                                 })
                             << "});\n";
                        for (auto it : axes)
                            fout << arg_names[curr] << " = " << arg_names[curr] << ".broadcast("
                                 << curr->get_output_shape(0)[it] << ", " << it << ");\n";
                    };

                    kernel_dict["Reshape"] = [&](std::shared_ptr<GNode>& curr,
                                                 std::ofstream& fout) {
                        auto _op = get_op_object<nnfusion::op::Reshape>(curr);
                        if (!_op->get_is_transpose())
                        {
                            fout << "Tensor " << arg_names[curr] << " = "
                                 << arg_names[curr->get_in_edge(0)->get_src()] << ".reshape({"
                                 << join_collections(
                                        curr->get_output_shape(0),
                                        [&](int idx, ssize_t val) { return std::to_string(val); })
                                 << "});\n";
                        }
                        else
                        {
                            fout << "Tensor " << arg_names[curr] << " = "
                                 << arg_names[curr->get_in_edge(0)->get_src()] << ".dimShuffle({"
                                 << join_collections(
                                        _op->get_input_order(),
                                        [](int idx, ssize_t val) { return std::to_string(val); })
                                 << "});\n";
                        }
                    };

                    kernel_dict["Concat"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        auto _op = get_op_object<nnfusion::op::Concat>(curr);

                        auto axis = _op->get_concatenation_axis();

                        fout << "Tensor " << arg_names[curr] << " = "
                             << arg_names[curr->get_in_edge(0)->get_src()] << ";\n";
                        for (int i = 1; i < curr->get_input_size(); ++i)
                            fout << arg_names[curr] << " = concat(" << arg_names[curr] << ", "
                                 << arg_names[curr->get_in_edge(1)->get_src()] << ", " << axis
                                 << ");\n";
                    };

                    kernel_dict["Slice"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        auto _op = get_op_object<nnfusion::op::Slice>(curr);

                        bool builtin_slice = true;
                        for (auto& it : _op->get_strides())
                            if (it != 1)
                            {
                                builtin_slice = false;
                                break;
                            }
                        if (builtin_slice)
                        {
                            fout << "Tensor " << arg_names[curr] << " = "
                                 << arg_names[curr->get_in_edge(0)->get_src()]
                                 << ".slice(ArrayRef<std::size_t>({"
                                 << join_collections(
                                        _op->get_lower_bounds(),
                                        [&](int idx, ssize_t val) { return std::to_string(val); })
                                 << "}), ArrayRef<std::size_t>({"
                                 << join_collections(
                                        _op->get_upper_bounds(),
                                        [&](int idx, ssize_t val) { return std::to_string(val); })
                                 << "}));\n";
                        }
                        else
                        {
                            UNHANDLED_CASE(curr);
                        }
                    };

                    kernel_dict["Dot"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        auto _op = get_op_object<nnfusion::op::Dot>(curr);

                        auto shape_0 = curr->get_input_shape(0);
                        auto shape_1 = curr->get_input_shape(1);
                        int N = shape_0[0], K = shape_0[1], M = shape_1[1];

                        if (getenv("GC_POPDOT") == nullptr && N == 1 && M <= max_threads)
                        {
                            NNFUSION_CHECK(_op->get_transpose_A() == false);

                            std::vector<std::string> convert_input(curr->get_input_size());
                            if (_op->get_transpose_B() == false)
                                convert_input[1] = ".transpose()";

                            auto code = autogen(op::create_code_from_template(
                                R"( - input("input0", @input_shape_0@); input("input1", @input_shape_1@); k = loop(@K@); output(@output_shape@, lambda i: tvm.te.sum(args("input0")[k] * args("input1")[k], axis=k)); )",
                                {{"input_shape_0", "[ " + std::to_string(K) + " ]"},
                                 {"input_shape_1", "[ " + std::to_string(K) + " ]"},
                                 {"output_shape", "[ 1 ]"},
                                 {"K", K}}));
                            print_standard_kernel_code(curr, fout, code, {1, M, M}, convert_input);
                        }
                        else
                        {
                            new_super_step();

                            assert(curr->get_output_element_type(0) == nnfusion::element::f32);

                            fout << op::create_code_from_template(
                                "Tensor @out_name@ = poplin::matMul(g, @A@, @B@, prog, "
                                "FLOAT);\n",
                                {
                                    {"out_name", arg_names[curr]},
                                    {"A",
                                     arg_names[curr->get_in_edge(0)->get_src()] +
                                         (_op->get_transpose_A() ? ".transpose()" : "")},
                                    {"B",
                                     arg_names[curr->get_in_edge(1)->get_src()] +
                                         (_op->get_transpose_B() ? ".transpose()" : "")},
                                });
                        }
                    };

                    kernel_dict["BatchMatMul"] = [&](std::shared_ptr<GNode>& curr,
                                                     std::ofstream& fout) {
                        auto generic_op = get_op_object<nnfusion::op::GenericOp>(curr);
                        bool transA = generic_op->localOpConfig.getRoot()["adj_x"]["b"];
                        bool transB = generic_op->localOpConfig.getRoot()["adj_y"]["b"];

                        auto shape_0 = curr->get_input_shape(0);
                        auto shape_1 = curr->get_input_shape(1);
                        auto out_shape = curr->get_output_shape(0);

                        NNFUSION_CHECK(shape_0.size() == shape_1.size());

                        int batch = 1;
                        for (int i = shape_0.size() - 3; i >= 0; --i)
                            batch *= shape_0[i];

                        new_super_step();

                        assert(curr->get_output_element_type(0) == nnfusion::element::f32);

                        fout << op::create_code_from_template(
                            "Tensor @out_name@ = poplin::matMulGrouped(g, "
                            "@A@.reshape({@batch@, @orginA..@})@transA@, "
                            "@B@.reshape({@batch@, @orginB..@})@transB@, prog, "
                            "FLOAT).reshape({@out_shape@});\n",
                            {
                                {"out_name", arg_names[curr]},
                                {"out_shape",
                                 join_collections(
                                     out_shape,
                                     [](int idx, ssize_t val) { return std::to_string(val); })},
                                {"batch", batch},
                                {"A", arg_names[curr->get_in_edge(0)->get_src()]},
                                {"B", arg_names[curr->get_in_edge(1)->get_src()]},
                                {"orginA..",
                                 std::to_string(shape_0[shape_0.size() - 2]) + ", " +
                                     std::to_string(shape_0[shape_0.size() - 1])},
                                {"orginB..",
                                 std::to_string(shape_1[shape_1.size() - 2]) + ", " +
                                     std::to_string(shape_1[shape_1.size() - 1])},
                                {"transA", transA ? ".dimShuffle({0, 2, 1})" : ""},
                                {"transB", transB ? ".dimShuffle({0, 2, 1})" : ""},
                            });
                    };

                    kernel_dict["Convert"] = [&](std::shared_ptr<GNode>& curr,
                                                 std::ofstream& fout) {
                        auto _op = get_op_object<nnfusion::op::Convert>(curr);
                        auto dtype = _op->get_convert_element_type();
                        std::string output_type;
                        if (dtype == nnfusion::element::f32)
                            output_type = "FLOAT";
                        else if (dtype == nnfusion::element::i32)
                            output_type = "UNSIGNED_INT";
                        else
                            assert(0);

                        new_super_step();
                        fout << "Tensor " << arg_names[curr] << " = popops::cast(g, "
                             << arg_names[curr->get_in_edge(0)->get_src()] << ", " << output_type
                             << ", prog);\n";
                    };

                    kernel_dict["OneHot"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        auto generic_op = get_op_object<nnfusion::op::GenericOp>(curr);
                        int depth = generic_op->localOpConfig.getRoot()["depth"];
                        float on_value = generic_op->localOpConfig.getRoot()["on_value"];
                        float off_value = generic_op->localOpConfig.getRoot()["off_value"];
                        int axis = generic_op->localOpConfig.getRoot()["axis"];
                        auto encoded = arg_names[curr];
                        auto indices = arg_names[curr->get_in_edge(0)->get_src()];
                        auto output_shape = curr->get_output_shape(0);
                        auto dtype = curr->get_output_element_type(0).c_type_string();
                        if (dtype == "float")
                            dtype = "FLOAT";

                        new_super_step();

                        fout << op::create_code_from_template(
                            "Tensor @encoded@ = g.addVariable(@dtype@, "
                            "poplar::ArrayRef<std::size_t>({@output_shape@}), \"@encoded@\");\n"
                            "Tensor @encoded@_on_val = g.addConstant<float>(FLOAT, {}, "
                            "{@on_val@});\n"
                            "Tensor @encoded@_off_val = g.addConstant<float>(FLOAT, {}, "
                            "{@off_val@});\n"
                            "place_tensor(g, @encoded@);\n"
                            "place_tensor(g, @encoded@_on_val);\n"
                            "place_tensor(g, @encoded@_off_val);\n"
                            "popops::encodeOneHot(g, @indices@, @encoded@, prog, "
                            "{@encoded@_on_val}, "
                            "{@encoded@_off_val});\n",
                            {
                                {"encoded", encoded},
                                {"dtype", dtype},
                                {"output_shape", join(output_shape)},
                                {"indices", indices},
                                {"on_val", on_value},
                                {"off_val", off_value},
                            });
                    };

                    kernel_dict["GatherV2"] = [&](std::shared_ptr<GNode>& curr,
                                                  std::ofstream& fout) {
                        auto generic_op = get_op_object<nnfusion::op::GenericOp>(curr);
                        int axis = generic_op->localOpConfig.getRoot()["axis"];

                        auto shape_0 = curr->get_input_shape(0);
                        auto shape_1 = curr->get_in_edge(1)->get_src()->get_output_shape(0);
                        int N = shape_0[0], K = shape_0[1], M = shape_1[1];

                        new_super_step();

                        assert(curr->get_output_element_type(0) == nnfusion::element::f32);

                        fout << op::create_code_from_template(
                            "Tensor @out_name@ = popops::gather(g, @A@, @B@, @axis@, "
                            "prog, popops::GatherParams());\n",
                            {{"out_name", arg_names[curr]},
                             {"A", arg_names[curr->get_in_edge(0)->get_src()]},
                             {"B", arg_names[curr->get_in_edge(1)->get_src()]},
                             {"axis", axis}});
                    };

                    kernel_dict["Convolution"] = [&](std::shared_ptr<GNode>& curr,
                                                     std::ofstream& fout) {
                        new_super_step();

                        assert(curr->get_output_element_type(0) == nnfusion::element::f32);

                        auto _op = get_op_object<nnfusion::op::Convolution>(curr);
                        for (auto& it : _op->get_data_dilation_strides())
                            NNFUSION_CHECK(it == 1);

                        auto data_shape = curr->get_input_shape(0);
                        auto weight_shape = curr->get_input_shape(1);
                        auto out_shape = curr->get_output_shape(0);

                        fout << op::create_code_from_template(
                            "Tensor @out_name@ = poplin::convolution(g, @data@, @weight@, "
                            "poplin::ConvParams(FLOAT, FLOAT, @N@, {@HI@, @WI@}, {@HK@, "
                            "@WK@}, @CI@, @CO@, 1, poplin::ConvParams::InputTransform({0, "
                            "0}, {0, 0}, {1, 1}, {@pad_lower_h@, @pad_lower_w@}, "
                            "{@pad_upper_h@, @pad_upper_w@}, {false, false}), "
                            "poplin::ConvParams::InputTransform(2), "
                            "poplin::ConvParams::OutputTransform({0, 0}, {0, 0}, "
                            "{@stride_h@, @stride_w@}, {0, 0}, {0, 0})), false, "
                            "prog).reshape({@out_shape@});\n",
                            {
                                {"out_name", arg_names[curr]},
                                {"out_shape",
                                 join_collections(
                                     out_shape,
                                     [](int idx, ssize_t val) { return std::to_string(val); })},
                                {"data", arg_names[curr->get_in_edge(0)->get_src()]},
                                {"weight", arg_names[curr->get_in_edge(1)->get_src()]},
                                {"N", data_shape[0]},
                                {"HI", data_shape[2]},
                                {"WI", data_shape[3]},
                                {"HK", weight_shape[2]},
                                {"WK", weight_shape[3]},
                                {"CI", data_shape[1]},
                                {"CO", weight_shape[0]},
                                {"pad_lower_h", _op->get_padding_below()[0]},
                                {"pad_lower_w", _op->get_padding_below()[1]},
                                {"pad_upper_h", _op->get_padding_above()[0]},
                                {"pad_upper_w", _op->get_padding_above()[1]},
                                {"stride_h", _op->get_window_movement_strides()[0]},
                                {"stride_w", _op->get_window_movement_strides()[1]},
                            });
                    };

                    kernel_dict["AvgPool"] = [&](std::shared_ptr<GNode>& curr,
                                                 std::ofstream& fout) {
                        new_super_step();

                        assert(curr->get_output_element_type(0) == nnfusion::element::f32);

                        auto _op = get_op_object<nnfusion::op::AvgPool>(curr);

                        bool use_padding = false;
                        for (auto& it : _op->get_padding_below())
                            if (it != 0)
                                use_padding = true;
                        for (auto& it : _op->get_padding_above())
                            if (it != 0)
                                use_padding = true;

                        if (use_padding)
                        {
                            // TODO: NNFUSION_CHECK(_op->get_include_padding_in_avg_computation() == true);

                            auto pad_lower = _op->get_padding_below();
                            auto pad_upper = _op->get_padding_above();

                            fout << "Tensor T0_" << arg_names[curr] << " = popops::pad(g, "
                                 << arg_names[curr->get_in_edge(0)->get_src()] << ", {"
                                 << join_collections(
                                        _op->get_padding_below(),
                                        [](int idx, ssize_t val) { return std::to_string(val); })
                                 << "}, {" << join_collections(_op->get_padding_below(),
                                                               [](int idx, ssize_t val) {
                                                                   return std::to_string(val);
                                                               })
                                 << "}, 0.0f);\n";
                        }
                        else
                        {
                            fout << "Tensor &T0_" << arg_names[curr] << " = "
                                 << arg_names[curr->get_in_edge(0)->get_src()] << ";";
                        }

                        auto data_shape = curr->get_input_shape(0);
                        auto win_shape = _op->get_window_shape();
                        auto mov_stride = _op->get_window_movement_strides();
                        auto out_shape = curr->get_output_shape(0);

                        NNFUSION_CHECK(data_shape.size() == 4);
                        NNFUSION_CHECK(win_shape.size() == 2);
                        NNFUSION_CHECK(mov_stride.size() == 2);
                        NNFUSION_CHECK(out_shape.size() == 4);

                        fout << "Tensor " << arg_names[curr]
                             << " = popnn::pooling::pool(g, "
                                "popnn::pooling::PoolParams(popnn::PoolingType::AVG, {"
                             << data_shape[2] << ", " << data_shape[3] << "}, {" << win_shape[0]
                             << "," << win_shape[1] << "}, {" << mov_stride[0] << ", "
                             << mov_stride[1] << "}, "
                             << "{0, 0}, {0, 0}, " << data_shape[1] << ", " << data_shape[0]
                             << ", FLOAT), T0_" << arg_names[curr] << ", prog).reshape({"
                             << join_collections(
                                    curr->get_output_shape(0),
                                    [&](int idx, ssize_t val) { return std::to_string(val); })
                             << "});\n";
                    };

                    kernel_dict["BatchNormInference"] = [&](std::shared_ptr<GNode>& curr,
                                                            std::ofstream& fout) {
                        new_super_step();
                        assert(curr->get_output_element_type(0) == nnfusion::element::f32);

                        // auto _op = get_op_object<nnfusion::op::BatchNormInference>(curr);

                        fout << "Tensor " << arg_names[curr] << " = popnn::bn::batchNormalise(g, "
                             << arg_names[curr->get_in_edge(2)->get_src()] << ", "
                             << arg_names[curr->get_in_edge(0)->get_src()] << ", "
                             << arg_names[curr->get_in_edge(1)->get_src()] << ", "
                             << arg_names[curr->get_in_edge(3)->get_src()] << ", "
                             << arg_names[curr->get_in_edge(4)->get_src()]
                             << ", prog).first.reshape({"
                             << join_collections(
                                    curr->get_output_shape(0),
                                    [&](int idx, ssize_t val) { return std::to_string(val); })
                             << "});\n";
                    };

                    kernel_dict["Pad"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        new_super_step();

                        assert(curr->get_output_element_type(0) == nnfusion::element::f32);

                        auto _op = get_op_object<nnfusion::op::Pad>(curr);
                        for (auto& it : _op->get_padding_interior())
                            NNFUSION_CHECK(it == 0);

                        float pad_value;
                        auto fill_const = curr->get_in_edge(1)->get_src();
                        if (fill_const->is_constant())
                        {
                            pad_value =
                                *(float*)get_op_object<op::Constant>(fill_const)->get_data_ptr();
                        }
                        else
                        {
                            // TODO: ought to be constant input, but not handled
                            UNHANDLED_CASE(curr);
                        }

                        auto pad_lower = _op->get_padding_below();
                        auto pad_upper = _op->get_padding_above();

                        fout << "Tensor " << arg_names[curr] << " = popops::pad(g, "
                             << arg_names[curr->get_in_edge(0)->get_src()] << ", {"
                             << join_collections(
                                    _op->get_padding_below(),
                                    [](int idx, ssize_t val) { return std::to_string(val); })
                             << "}, {" << join_collections(_op->get_padding_below(),
                                                           [](int idx, ssize_t val) {
                                                               return std::to_string(val);
                                                           })
                             << "}, " << pad_value << ").reshape({"
                             << join_collections(
                                    curr->get_output_shape(0),
                                    [&](int idx, ssize_t val) { return std::to_string(val); })
                             << "});\n";
                    };

                    kernel_dict["Sum"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        auto _op = get_op_object<nnfusion::op::Sum>(curr);
                        auto axes = _op->get_reduction_axes();

                        auto input_shape = curr->get_input_shape(0);
                        int min_axis = INT_MAX;
                        if (axes.size() == 0)
                            min_axis = 0;
                        else
                            for (auto& axis : axes)
                                min_axis = min(min_axis, (int)axis);
                        if (input_shape.size() - axes.size() == min_axis || axes.size() == 0)
                        {
                            int batch = 1, sample = 1;
                            for (int i = 0; i < min_axis; ++i)
                                batch *= input_shape[i];
                            for (int i = min_axis; i < input_shape.size(); ++i)
                                sample *= input_shape[i];

                            auto code = autogen(op::create_code_from_template(
                                "- input(\"input0\", [@sample@]); output([1], "
                                "topi=topi.sum(args(\"input0\"), axis=0, keepdims=True));",
                                {{"sample", sample}}));
                            print_standard_kernel_code(
                                curr,
                                fout,
                                code,
                                std::vector<int>(1 + curr->get_input_size(), batch),
                                {});
                        }
                        else
                        {
                            new_super_step();

                            fout << "Tensor " << arg_names[curr] << " = popops::reduce(g, "
                                 << arg_names[curr->get_in_edge(0)->get_src()] << ", {"
                                 << join_collections(
                                        axes,
                                        [](int idx, ssize_t val) { return std::to_string(val); })
                                 << "}, popops::ReduceParams(popops::Operation::ADD), "
                                    "prog).reshape({"
                                 << join_collections(
                                        curr->get_output_shape(0),
                                        [&](int idx, ssize_t val) { return std::to_string(val); })
                                 << "});\n";
                        }
                    };

                    kernel_dict["Softmax"] = [&](std::shared_ptr<GNode>& curr,
                                                 std::ofstream& fout) {
                        new_super_step();

                        assert(curr->get_output_element_type(0) == nnfusion::element::f32);

                        auto _op = get_op_object<nnfusion::op::Softmax>(curr);
                        auto axes = _op->get_axes();
                        auto data_shape = curr->get_input_shape(0);
                        int groups = 1, sample_size = 1;
                        for (int i = 0; i < axes.size(); ++i)
                        {
                            NNFUSION_CHECK(axes.count(data_shape.size() - 1 - i));
                            sample_size *= data_shape[data_shape.size() - 1 - i];
                        }
                        for (int i = 0; i < data_shape.size() - axes.size(); ++i)
                            groups *= data_shape[i];

                        bool use_builtin = false;
                        if (use_builtin)
                        {
                            fout << "Tensor " << arg_names[curr]
                                 << " = popnn::spatialSoftMax2D(g, prog, "
                                 << arg_names[curr->get_in_edge(0)->get_src()] << ".reshape({"
                                 << groups << ", " << sample_size << ", 1}), 1.0f, false).first;\n";
                        }
                        else
                        {
                            fout << "Tensor T0_" << arg_names[curr]
                                 << " = popops::map(g, "
                                    "popops::expr::Exp(popops::expr::_1), {"
                                 << arg_names[curr->get_in_edge(0)->get_src()]
                                 << "}, prog).reshape({" << groups << ", " << sample_size
                                 << "});\n";

                            fout << "Tensor T1_" << arg_names[curr] << " = popops::reduce(g, "
                                 << "T0_" << arg_names[curr]
                                 << ", {1}, popops::ReduceParams(popops::Operation::ADD), "
                                    "prog).reshape({"
                                 << groups << ", 1}).broadcast(" << sample_size << ", 1);\n";

                            fout << "Tensor " << arg_names[curr]
                                 << " = popops::map(g, "
                                    "popops::expr::Divide(popops::expr::_1, "
                                    "popops::expr::_2), {"
                                 << "T0_" << arg_names[curr] << ", T1_" << arg_names[curr]
                                 << "}, prog);\n";
                        }
                    };

                    kernel_dict["DepthwiseConv2dNative"] = [&](std::shared_ptr<GNode>& curr,
                                                               std::ofstream& fout) {
                        new_super_step();

                        assert(curr->get_output_element_type(0) == nnfusion::element::f32);

                        auto _op = get_op_object<nnfusion::op::GenericOp>(curr);
                        auto& cfg = _op->localOpConfig.getRoot();

                        NNFUSION_CHECK(cfg["padding_type"] == "SAME");
                        bool channel_last = (cfg["data_format"] == "NHWC");

                        for (auto& it : cfg["dilations"])
                            NNFUSION_CHECK(it == 1);

                        auto data_shape = curr->get_input_shape(0);   // NHWC -> NCHW
                        auto weight_shape = curr->get_input_shape(1); // KKCF -> CF1KK
                        auto out_shape = curr->get_output_shape(0);   // NHW(FxC)

                        fout << op::create_code_from_template(
                            "Tensor @out_name@ = poplin::convolution(g, "
                            "@data@.dimShuffle({0, 3, 1, 2}), "
                            "@weight@.dimShuffle({2, 3, 0, 1}).reshape({@CI@, @CO@, 1, "
                            "@HK@, @WK@}), "
                            "poplin::ConvParams(FLOAT, FLOAT, @N@, {@HI@, @WI@}, {@HK@, "
                            "@WK@}, 1, @CO@, @CI@, poplin::ConvParams::InputTransform({0, "
                            "0}, {0, 0}, {1, 1}, {@pad_lower_h@, @pad_lower_w@}, "
                            "{@pad_upper_h@, @pad_upper_w@}, {false, false}), "
                            "poplin::ConvParams::InputTransform(2), "
                            "poplin::ConvParams::OutputTransform({0, 0}, {0, 0}, "
                            "{@stride_h@, @stride_w@}, {0, 0}, {0, 0})), false, "
                            "prog).dimShuffle({0, 2, 3, 1}).reshape({@out_shape@});\n",
                            {
                                {"out_name", arg_names[curr]},
                                {"out_shape",
                                 join_collections(
                                     out_shape,
                                     [](int idx, ssize_t val) { return std::to_string(val); })},
                                {"data", arg_names[curr->get_in_edge(0)->get_src()]},
                                {"weight", arg_names[curr->get_in_edge(1)->get_src()]},
                                {"N", data_shape[0]},
                                {"HI", channel_last ? data_shape[1] : data_shape[2]},
                                {"WI", channel_last ? data_shape[2] : data_shape[3]},
                                {"HK", weight_shape[0]},
                                {"WK", weight_shape[1]},
                                {"CI", channel_last ? data_shape[3] : data_shape[1]},
                                {"CO", weight_shape[3]},
                                {"pad_lower_h", cfg["padding_before"][0]},
                                {"pad_lower_w", cfg["padding_before"][1]},
                                {"pad_upper_h", cfg["padding_after"][0]},
                                {"pad_upper_w", cfg["padding_after"][1]},
                                {"stride_h", cfg["strides"][0]},
                                {"stride_w", cfg["strides"][1]},
                            });
                    };

                    kernel_dict["ApplyGradient"] = [&](std::shared_ptr<GNode>& curr,
                                                       std::ofstream& fout) {
                        auto _op = get_op_object<nnfusion::op::GenericOp>(curr);
                        auto& cfg = _op->localOpConfig.getRoot();
                        float lr =
                            cfg["learning_rate"].is_null() ? 0.001 : (float)cfg["learning_rate"];
                        codegen_for_elementwise(
                            curr,
                            fout,
                            "lambda x: args(\"input0\")[x] - args(\"input1\")[x] * " +
                                std::to_string(lr));
                    };

                    kernel_dict["DivNoNan"] = [&](std::shared_ptr<GNode>& curr,
                                                  std::ofstream& fout) {
                        codegen_for_elementwise(
                            curr,
                            fout,
                            "lambda x: tvm.te.if_then_else(args(\"input1\")[x] != "
                            "0, args(\"input0\")[x] / args(\"input1\")[x], 0)");
                    };

                    kernel_dict["ReluBackprop"] = [&](std::shared_ptr<GNode>& curr,
                                                      std::ofstream& fout) {
                        codegen_for_elementwise(
                            curr,
                            fout,
                            "lambda x: tvm.te.if_then_else(args(\"input0\")[x] > "
                            "0, args(\"input1\")[x], 0)");
                    };

                    kernel_dict["Select"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        auto output = arg_names[curr];
                        auto input_a = arg_names[curr->get_in_edge(0)->get_src()];
                        auto input_b = arg_names[curr->get_in_edge(1)->get_src()];
                        auto input_c = arg_names[curr->get_in_edge(2)->get_src()];
                        auto dtype = curr->get_output_element_type(0).c_type_string();
                        auto output_shape = curr->get_output_shape(0);
                        new_super_step();

                        fout << op::create_code_from_template(
                            "Tensor @output@ = popops::map(g, "
                            "popops::expr::TernaryOp(popops::expr::TernaryOpType::SELECT, "
                            "popops::expr::_1, popops::expr::_2, popops::expr::_3), {@input_a@, "
                            "@input_b@, @input_c@}, prog);\n",
                            {
                                {"output", output},
                                {"input_a", input_a},
                                {"input_b", input_b},
                                {"input_c", input_c},
                            });
                    };

                    /*
                    kernel_dict["Tile"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        nnfusion::Shape input_shape = curr->get_input_shape(0);
                        nnfusion::Shape output_shape = curr->get_output_shape(0);

                        auto ng_op = curr->get_in_edge(1)->get_src();
                        NNFUSION_CHECK(ng_op->is_constant())
                            << "We only accept the Tile input \"multiples\" as Constant.";
                        ///\todo multiples must be int32 or int64, we use int32 in this case, currently we ignore int64
                        auto multiples =
                            std::dynamic_pointer_cast<nnfusion::op::Constant>(ng_op->get_op_ptr())
                                ->get_vector<int64_t>();

                        auto expression = op::create_code_from_template(
                            R"( - input("input0", @input_shape@); output(@output_shape@, topi=topi.tile(args("input0"), @multiples@)); )",
                            {{"input_shape", vector_to_string(input_shape)},
                             {"output_shape", vector_to_string(output_shape)},
                             {"multiples", vector_to_string(multiples)}});

                        auto code = autogen(expression);

                        int num_elements = 1, y;
                        for (auto& it : curr->get_output_shape(0))
                            num_elements *= it;
                        for (int i = max_threads; i >= 1; --i)
                            if (num_elements % i == 0)
                            {
                                y = i;
                                break;
                            }

                        print_standard_kernel_code(
                            curr, fout, code, std::vector<int>(1 + curr->get_input_size(), y), {});
                    };
                    */

                    kernel_dict["MaxPool"] = [&](std::shared_ptr<GNode>& curr,
                                                 std::ofstream& fout) {
                        new_super_step();

                        assert(curr->get_output_element_type(0) == nnfusion::element::f32);

                        auto _op = get_op_object<nnfusion::op::MaxPool>(curr);
                        bool use_padding = false;
                        for (auto& it : _op->get_padding_below())
                            if (it != 0)
                                use_padding = true;
                        for (auto& it : _op->get_padding_above())
                            if (it != 0)
                                use_padding = true;

                        if (use_padding)
                        {
                            auto pad_lower = _op->get_padding_below();
                            auto pad_upper = _op->get_padding_above();

                            fout << "Tensor T0_" << arg_names[curr] << " = popops::pad(g, "
                                 << arg_names[curr->get_in_edge(0)->get_src()] << ", {"
                                 << join_collections(
                                        _op->get_padding_below(),
                                        [](int idx, ssize_t val) { return std::to_string(val); })
                                 << "}, {" << join_collections(_op->get_padding_below(),
                                                               [](int idx, ssize_t val) {
                                                                   return std::to_string(val);
                                                               })
                                 << "}, 0.0f);\n";
                        }
                        else
                        {
                            fout << "Tensor &T0_" << arg_names[curr] << " = "
                                 << arg_names[curr->get_in_edge(0)->get_src()] << ";";
                        }

                        auto data_shape = curr->get_input_shape(0);
                        auto win_shape = _op->get_window_shape();
                        auto mov_stride = _op->get_window_movement_strides();
                        auto out_shape = curr->get_output_shape(0);

                        NNFUSION_CHECK(data_shape.size() == 4);
                        NNFUSION_CHECK(win_shape.size() == 2);
                        NNFUSION_CHECK(mov_stride.size() == 2);
                        NNFUSION_CHECK(out_shape.size() == 4);

                        fout << "Tensor " << arg_names[curr]
                             << " = popnn::pooling::pool(g, "
                                "popnn::pooling::PoolParams(popnn::PoolingType::MAX, {"
                             << data_shape[2] << ", " << data_shape[3] << "}, {" << win_shape[0]
                             << "," << win_shape[1] << "}, {" << mov_stride[0] << ", "
                             << mov_stride[1] << "}, "
                             << "{0, 0}, {0, 0}, " << data_shape[1] << ", " << data_shape[0]
                             << ", FLOAT), T0_" << arg_names[curr] << ", prog).reshape({"
                             << join_collections(
                                    curr->get_output_shape(0),
                                    [&](int idx, ssize_t val) { return std::to_string(val); })
                             << "});\n";
                    };

                    while (gen_q.size() > 0 || pend_q.size() > 0)
                    {
                        // Move to new super step if satisifed
                        if (!gen_q.size())
                            new_super_step();

                        auto curr = gen_q.front();
                        gen_q.pop_front();
                        visited.insert(curr);

                        // fout << "DEBUG(\"" << arg_names[curr] << "\");\n";

                        auto entry = kernel_dict.find(curr->get_op_ptr()->get_op_type());
                        if (entry != kernel_dict.end())
                            entry->second(curr, fout);
                        else
                        {
                            UNHANDLED_CASE(curr);
                        }
                        fout << std::endl;

                        // Check its children about whether all inputs are ready (Must be put after any possible new_super_step())
                        for (auto& edge : curr->get_out_edges())
                        {
                            if (edge->is_control_edge())
                                continue;
                            NNFUSION_CHECK(edge->get_src() == curr);
                            NNFUSION_CHECK(visited.count(edge->get_dst()) == 0);

                            bool ready = true;
                            for (auto& from : edge->get_dst()->get_in_edges())
                            {
                                if (from->is_control_edge())
                                    continue;
                                if (visited.count(from->get_src()) == 0)
                                {
                                    ready = false;
                                    break;
                                }
                            }
                            if (ready)
                            {
                                // Only join pend_q once
                                if (vis_pend.count(edge->get_dst()) == 0)
                                {
                                    vis_pend.insert(edge->get_dst());
                                    pend_q.push_back(edge->get_dst());
                                }
                            }
                        }
                    }

                    // Print Results
                    for (auto& curr : graph->get_outputs()) // Print output nodes
                    // for (auto& curr : graph->get_nodes()) // Print all nodes
                    {
                        if (blacklist.count(curr))
                            continue;
                        fout << "print_tensor(\"Result(" << arg_names[curr] << ")\", "
                             << arg_names[curr] << ");\n\n";
                    }

                    fout << "#endif" << std::endl;

                    nnfusion::codegen::copy_file_from_templates(currentBackend + "/Makefile",
                                                                "nnfusion_rt/" + currentBackend +
                                                                    "_codegen/Makefile");
                    nnfusion::codegen::copy_file_from_templates(currentBackend + "/run_graph.cpp",
                                                                "nnfusion_rt/" + currentBackend +
                                                                    "_codegen/run_graph.cpp");
                    nnfusion::codegen::copy_file_from_templates(currentBackend + "/picosha2.h",
                                                                "nnfusion_rt/" + currentBackend +
                                                                    "_codegen/picosha2.h");
                    NNFUSION_LOG(INFO) << currentBackend << " codegen finished.";
                    exit(0);
                    return true;
                }
            };
        } // namespace graph
    }     // namespace pass
} // namespace nnfusion
