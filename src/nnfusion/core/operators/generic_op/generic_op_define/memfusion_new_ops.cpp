#include "nnfusion/core/operators/generic_op/generic_op.hpp"

static string make_layout(const std::set<int>& axes)
{
    std::string ret = "";
    for (auto ax : axes)
        ret += ", N" + std::to_string(ax);
    return "[" + (axes.empty() ? "N" : ret.substr(2)) + "]";
};

REGISTER_OP(SoftmaxBasic)
    .attr<vector<int>>("axes")
    .attr<int>("stage")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto& shape_0 = gnode->get_input_shape(0);
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        vector<int> axes = generic_op->localOpConfig.getRoot()["axes"];
        int stage = generic_op->localOpConfig.getRoot()["stage"];
        nnfusion::Shape output_shape;
        if (stage == 1 || stage == 3)
        {
            output_shape = shape_0;
        }
        else
        {
            set<int> ax_set(axes.begin(), axes.end());
            for (int i = 0; i < shape_0.size(); i++)
            {
                if (ax_set.count(i))
                    continue;
                output_shape.push_back(shape_0[i]);
            }
        }
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        std::set<int> input_ax, output_ax;
        auto input_shape = curr->get_input_shape(0);
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        vector<int> axes = generic_op->localOpConfig.getRoot()["axes"];
        int stage = generic_op->localOpConfig.getRoot()["stage"];
        set<int> ax_set(axes.begin(), axes.end());
        for (int i = 0; i < input_shape.size(); ++i)
        {
            if (!ax_set.count(i))
                output_ax.insert(i);
            input_ax.insert(i);
        }
        string expression_template;
        if (stage == 0)
        {
            expression_template =
                R"( @output0@@temp_layout@ >=! @input0@@input0_layout@; )";
        }
        else if (stage == 1)
        {
            expression_template =
                R"( @output0@@input0_layout@ = (@input0@@input0_layout@ - @input1@@temp_layout@).call(`exp`); )";
        }
        else if (stage == 2)
        {
            expression_template =
                R"( @output0@@temp_layout@ +=! @input0@@input0_layout@; )";
        }
        else if (stage == 3)
        {
            expression_template =
                R"( @output0@@input0_layout@ = @input0@@input0_layout@ / @input1@@temp_layout@; )";
        }
        else
        {
            NNFUSION_CHECK_FAIL() << "Incorrect Stage ID.";
        }
        std::string expression_code = op::create_code_from_template(
            expression_template,
            {{"temp_layout", make_layout(output_ax)}, {"input0_layout", make_layout(input_ax)}});
        return expression_code;
    });

REGISTER_OP(CNHW2NCHW)
    .attr<size_t>("N")
    .attr<size_t>("C")
    .attr<size_t>("H")
    .attr<size_t>("W")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        size_t N = generic_op->localOpConfig.getRoot()["N"];
        size_t C = generic_op->localOpConfig.getRoot()["C"];
        size_t H = generic_op->localOpConfig.getRoot()["H"];
        size_t W = generic_op->localOpConfig.getRoot()["W"];
        nnfusion::Shape output_shape{N, C, H, W};
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        string expression_template =
            R"( @output0@[N, C, H, W] = @input0@[C, W+H*@W@+N*@H*W@] where N in @N@, H in @H@, W in @W@; )";
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        size_t H = generic_op->localOpConfig.getRoot()["H"];
        size_t W = generic_op->localOpConfig.getRoot()["W"];
        size_t N = generic_op->localOpConfig.getRoot()["N"];
        nnfusion::json config;
        config["W"] = W;
        config["H"] = H;
        config["N"] = N;
        config["H*W"] = H * W;
        string expression_code = op::create_code_from_template(expression_template, config);
        return expression_code;
    });

REGISTER_OP(ImplicitGemm)
    .attr<size_t>("N")
    .attr<size_t>("C")
    .attr<size_t>("H")
    .attr<size_t>("W")
    .attr<size_t>("P")
    .attr<size_t>("S")
    .attr<size_t>("D")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        size_t c = generic_op->localOpConfig.getRoot()["C"];
        size_t n = generic_op->localOpConfig.getRoot()["N"];
        size_t h = generic_op->localOpConfig.getRoot()["H"];
        size_t w = generic_op->localOpConfig.getRoot()["W"];
        nnfusion::Shape output_shape{c, n * h * w};
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        size_t kh = curr->get_input_shape(1)[2];
        size_t kw = curr->get_input_shape(1)[3];
        size_t n = curr->get_input_shape(0)[0];
        size_t c = curr->get_input_shape(0)[1];
        size_t inh = curr->get_input_shape(0)[2];
        size_t inw = curr->get_input_shape(0)[3];
        size_t f = generic_op->localOpConfig.getRoot()["C"];
        size_t h = generic_op->localOpConfig.getRoot()["H"];
        size_t w = generic_op->localOpConfig.getRoot()["W"];
        size_t p = generic_op->localOpConfig.getRoot()["P"];
        size_t s = generic_op->localOpConfig.getRoot()["S"];
        size_t d = generic_op->localOpConfig.getRoot()["D"];
        NNFUSION_CHECK(inh = (h - 1) * s + (kh - 1) * d + 1 - 2 * p);
        NNFUSION_CHECK(inw = (w - 1) * s + (kw - 1) * d + 1 - 2 * p);
        size_t padh = inh + 2 * p, padw = inw + 2 * p;
        string pad_template = "";
        string data_template =
            R"( data[K, N] = @input0@[N//@h*w@, K//@kh*kw@, N%@h*w@//@w@*@s@+K%@kh*kw@//@kw@*@d@, N%@w@*@s@+K%@kw@*@d@] where K in @kh*kw*c@, N in @n*h*w@; )";
        string kernel_template =
            R"( kernel[M, K] = @input1@[M, K//@kh*kw@, K%@kh*kw@//@kw@, K%@kw@] where K in @kh*kw*c@, M in @f@; )";
        string compute_template = R"( @output0@[M, N] +=! kernel[M, K] * data[K, N]; )";
        if (p != 0)
        {
            pad_template =
                R"( pad[N, C, H0, W0] = @input0@[N, C, H0-@p@, W0-@p@].when([H0>=@p@, H0<@inh+p@, W0>=@p@, W0<@inw+p@], const(0.0).cast(input0[N, C, H0-@p@, W0-@p@].dtype())) where H0 in @padh@, W0 in @padw@; )";
            string input_str = "@input0@";
            data_template.replace(data_template.find(input_str), input_str.size(), "pad");
        }
        string expression_template =
            pad_template + data_template + kernel_template + compute_template;
        nnfusion::json config;
        config["p"] = p;
        config["s"] = s;
        config["d"] = d;
        config["padw"] = inh + 2 * p;
        config["padh"] = inw + 2 * p;
        config["inh+p"] = inh + p;
        config["inw+p"] = inw + p;
        config["w"] = w;
        config["h*w"] = h * w;
        config["kh*kw"] = kh * kw;
        config["kw"] = kw;
        config["kh*kw*c"] = kh * kw * c;
        config["n*h*w"] = n * h * w;
        config["f"] = f;
        string ir = op::create_code_from_template(expression_template, config);
        if (curr->get_output_element_type(0) == nnfusion::element::f16)
        {
            ir += "## @: tensorCoreConfig=(0, 1)";
        }
        return ir;
    });

REGISTER_OP(FusedDot)
    .attr<size_t>("M")
    .attr<size_t>("N")
    .attr<bool>("transpose_A")
    .attr<bool>("transpose_B")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        size_t m = generic_op->localOpConfig.getRoot()["M"];
        size_t n = generic_op->localOpConfig.getRoot()["N"];
        nnfusion::Shape output_shape{m, n};
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        string fuse_template =
            R"( temp0@A_fused_layout@ +=! @input0@@A_layout@ where M in @M@;)";
        string compute_template =
            R"( @output0@[M, N] +=! temp0@A_fused_layout@ * @input1@@B_layout@; )";
        string ir_template = fuse_template + compute_template;
        bool transpose_A = generic_op->localOpConfig.getRoot()["transpose_A"];
        bool transpose_B = generic_op->localOpConfig.getRoot()["transpose_B"];
        size_t m = generic_op->localOpConfig.getRoot()["M"];
        op::OpConfig::any op_config;
        op_config["M"] = m;
        op_config["A_fused_layout"] = transpose_A ? "[K, M]" : "[M, K]";
        op_config["B_layout"] = transpose_B ? "[N, K]" : "[K, N]";

        auto A_shape = curr->get_input_shape(0);
        int raxis = transpose_A ? A_shape.size() - 2 : A_shape.size() - 1;
        string A_layout;
        size_t stride = m;
        for (int i = 0; i < A_shape.size(); i++)
        {
            if (i > 0)
                A_layout += ", ";
            if (i == raxis)
                A_layout += "K";
            else
            {
                stride /= A_shape[i];
                A_layout += "M//" + to_string(stride) + "%" + to_string(A_shape[i]);
            }
        }
        op_config["A_layout"] = "[" + A_layout + "]";

        auto ir = op::create_code_from_template(ir_template, op_config);

        if (curr->get_output_element_type(0) == nnfusion::element::f16)
        {
            ir += "## @: tensorCoreConfig=(0, 1)";
        }
        return ir;
    });

REGISTER_OP(Conv1DImplicitGemm)
    .attr<size_t>("N")
    .attr<size_t>("C")
    .attr<size_t>("L")
    .attr<size_t>("P")
    .attr<size_t>("S")
    .attr<size_t>("D")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        size_t c = generic_op->localOpConfig.getRoot()["C"];
        size_t n = generic_op->localOpConfig.getRoot()["N"];
        size_t l = generic_op->localOpConfig.getRoot()["L"];
        nnfusion::Shape output_shape{c, n * l};
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        // N, C, L
        // F, C, KL
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        size_t kl = curr->get_input_shape(1)[2];
        size_t n = curr->get_input_shape(0)[0];
        size_t c = curr->get_input_shape(0)[1];
        size_t inl = curr->get_input_shape(0)[2];
        size_t f = generic_op->localOpConfig.getRoot()["C"];
        size_t l = generic_op->localOpConfig.getRoot()["L"];
        size_t p = generic_op->localOpConfig.getRoot()["P"];
        size_t s = generic_op->localOpConfig.getRoot()["S"];
        size_t d = generic_op->localOpConfig.getRoot()["D"];
        NNFUSION_CHECK(inl = (l - 1) * s + (kl - 1) * d + 1 - 2 * p);
        size_t padl = inl + 2 * p;
        string pad_template = "";
        string data_template =
            R"( data[K, N] = @input0@[N//@l@, K//@kl@, N%@l@*@s@+K%@kl@*@d@] where K in @kl*c@, N in @n*l@; )";
        string kernel_template =
            R"( kernel[M, K] = @input1@[M, K//@kl@, K%@kl@] where K in @kl*c@, M in @f@; )";
        string compute_template = R"( @output0@[M, N] +=! kernel[M, K] * data[K, N]; )";
        if (p != 0)
        {
            pad_template =
                R"( pad[N, C, L0] = @input0@[N, C, L0-@p@].when([L0>=@p@, L0<@inl+p@], const(0.0).cast(@input0@[N, C, L0-@p@].dtype())) where L0 in @padl@; )";
            string input_str = "@input0@";
            data_template.replace(data_template.find(input_str), input_str.size(), "pad");
        }
        string expression_template =
            pad_template + data_template + kernel_template + compute_template;
        nnfusion::json config;
        config["p"] = p;
        config["s"] = s;
        config["d"] = d;
        config["padl"] = inl + 2 * p;
        config["inl+p"] = inl + p;
        config["l"] = l;
        config["kl"] = kl;
        config["kl*c"] = kl * c;
        config["n*l"] = n * l;
        config["f"] = f;
        string ir = op::create_code_from_template(expression_template, config);
        if (curr->get_output_element_type(0) == nnfusion::element::f16)
        {
            ir += "## @: tensorCoreConfig=(0, 1)";
        }
        return ir;
    });

REGISTER_OP(CNW2NCW)
    .attr<size_t>("N")
    .attr<size_t>("C")
    .attr<size_t>("L")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        size_t N = generic_op->localOpConfig.getRoot()["N"];
        size_t C = generic_op->localOpConfig.getRoot()["C"];
        size_t L = generic_op->localOpConfig.getRoot()["L"];
        nnfusion::Shape output_shape{N, C, L};
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        string expression_template =
            R"( @output0@[N, C, L] = @input0@[C, L+N*@L@] where N in @N@, L in @L@; )";
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        size_t L = generic_op->localOpConfig.getRoot()["L"];
        size_t N = generic_op->localOpConfig.getRoot()["N"];
        nnfusion::json config;
        config["L"] = L;
        config["N"] = N;
        string expression_code = op::create_code_from_template(expression_template, config);
        return expression_code;
    });

REGISTER_OP(HardSigmoid)
    .attr<float>("alpha")
    .attr<float>("beta")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(1 == gnode->get_input_size());
        gnode->set_output_type_and_shape(
            0, gnode->get_input_element_type(0), gnode->get_input_shape(0));
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto ir_template =
            R"( @output0@@layout@ = (@input0@@layout@ * const(@alpha@).cast(@dtype@) + const(@beta@).cast(@dtype@)).call(`max`, [const(0).cast(@dtype@)]).call(`min`, [const(1).cast(@dtype@)]); )";
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        auto input0_shape = nnfusion::Shape(curr->get_input_shape(0));
        float alpha = generic_op->localOpConfig.getRoot()["alpha"];
        float beta = generic_op->localOpConfig.getRoot()["beta"];

        op::OpConfig::any op_config;
        set<int> axes;
        for (int i = 0; i < input0_shape.size(); i++)
            axes.insert(i);
        op_config["layout"] = make_layout(axes);
        op_config["alpha"] = std::to_string(alpha);
        op_config["beta"] = std::to_string(beta);
        ;
        string dtype;
        NNFUSION_CHECK(
            element::Type::nnfusion_element_type_to_dtype_string(curr->get_element_type(), dtype));
        op_config["dtype"] = "`" + dtype + "`";

        return op::create_code_from_template(ir_template, op_config);
    });

REGISTER_OP(Permutate)
    .attr<int>("type", 0)
    .attr<int>("inner_i", 16)
    .attr<int>("inner_j", 16)
    .infershape(
        [](std::shared_ptr<graph::GNode> gnode) -> void
        {
            NNFUSION_CHECK(1 == gnode->get_input_size());
            gnode->set_output_type_and_shape(
                0, gnode->get_input_element_type(0), gnode->get_input_shape(0));
        })
    .translate_v2(
        [](std::shared_ptr<graph::GNode> curr) -> std::string
        {
            // create expression `mediate0[N0, N1] = input0[N0 // 512 , N0 % 512, N1] where N0 in 16384;output0[N0, N1, N2, N3] = mediate0[(N0 * 16 + N2) // 16 * 16 + (N0 * 16 + N2)  % 8 * 2 + (N1 * 16 + N3) % 16 // 8, (N1 * 16 + N3) // 16 * 16 + (N0 * 16 + N2)  % 16 // 8 * 8 + (N1 * 16 + N3) % 8] where N0 in 1024, N1 in 256, N2 in 16, N3 in 16;`
            auto generic_op =
                std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
            auto input0_shape = nnfusion::Shape(curr->get_input_shape(0));
            auto input0_type = curr->get_input_element_type(0);
            NNFUSION_CHECK(input0_shape.size() == 2 || input0_shape.size() == 3)
                << "Currently only support 2D or 3D input";
            int type = generic_op->localOpConfig.getRoot()["type"];
            string expression_template;
            string expression_code;
            if (input0_shape.size() == 2)
            {
                if (type == 0)
                {
                    expression_template =
                        R"(@output0@[N0, N1, N2, N3] = @input0@[(N0 * 16 + N2) // 16 * 16 + (N0 * 16 + N2)  % 8 * 2 + (N1 * 16 + N3) % 16 // 8, (N1 * 16 + N3) // 16 * 16 + (N0 * 16 + N2)  % 16 // 8 * 8 + (N1 * 16 + N3) % 8] where N0 in @N0@, N1 in @N1@, N2 in @N2@, N3 in @N3@;)";
                }
                else
                {
                    NNFUSION_CHECK_FAIL() << "Permutate type not supported";
                }
                nnfusion::json config;
                config["N0"] = input0_shape[0] /
                               static_cast<int>(generic_op->localOpConfig.getRoot()["inner_i"]);
                config["N1"] = input0_shape[1] /
                               static_cast<int>(generic_op->localOpConfig.getRoot()["inner_j"]);
                config["N2"] = generic_op->localOpConfig.getRoot()["inner_i"];
                config["N3"] = generic_op->localOpConfig.getRoot()["inner_j"];
                expression_code = op::create_code_from_template(expression_template, config);
            }
            else if (input0_shape.size() == 3)
            {
                if (type == 0)
                {
                    expression_template =
                        R"( mediate0[N0, N1] = @input0@[N0 // 512 , N0 % 512, N1] where N0 in @M@;@output0@[N0, N1, N2, N3] = mediate0[(N0 * 16 + N2) // 16 * 16 + (N0 * 16 + N2)  % 8 * 2 + (N1 * 16 + N3) % 16 // 8, (N1 * 16 + N3) // 16 * 16 + (N0 * 16 + N2)  % 16 // 8 * 8 + (N1 * 16 + N3) % 8] where N0 in @N0@, N1 in @N1@, N2 in @N2@, N3 in @N3@;)";
                }
                else
                {
                    NNFUSION_CHECK_FAIL() << "Permutate type not supported";
                }

                nnfusion::json config;
                config["M"] = input0_shape[0] * input0_shape[1];
                config["N0"] = (input0_shape[0] * input0_shape[1]) /
                               static_cast<int>(generic_op->localOpConfig.getRoot()["inner_i"]);
                config["N1"] = input0_shape[2] /
                               static_cast<int>(generic_op->localOpConfig.getRoot()["inner_j"]);
                config["N2"] = generic_op->localOpConfig.getRoot()["inner_i"];
                config["N3"] = generic_op->localOpConfig.getRoot()["inner_j"];
                expression_code = op::create_code_from_template(expression_template, config);
            }
            return expression_code;
        });

REGISTER_OP(BatchPermutate)
    .attr<int>("batch_dims", 2)
    .attr<int>("type", 0)
    .attr<int>("inner_i", 16)
    .attr<int>("inner_j", 16)
    .infershape(
        [](std::shared_ptr<graph::GNode> gnode) -> void
        {
            NNFUSION_CHECK(1 == gnode->get_input_size());
            gnode->set_output_type_and_shape(
                0, gnode->get_input_element_type(0), gnode->get_input_shape(0));
        })
    .translate_v2(
        [](std::shared_ptr<graph::GNode> curr) -> std::string
        {
            // create expression `mediate0[N0, N1] = input0[N0 // 512 , N0 % 512, N1] where N0 in 16384;output0[N0, N1, N2, N3] = mediate0[(N0 * 16 + N2) // 16 * 16 + (N0 * 16 + N2)  % 8 * 2 + (N1 * 16 + N3) % 16 // 8, (N1 * 16 + N3) // 16 * 16 + (N0 * 16 + N2)  % 16 // 8 * 8 + (N1 * 16 + N3) % 8] where N0 in 1024, N1 in 256, N2 in 16, N3 in 16;`
            auto generic_op =
                std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
            auto input0_shape = nnfusion::Shape(curr->get_input_shape(0));
            auto input0_type = curr->get_input_element_type(0);
            NNFUSION_CHECK(input0_shape.size() == 3 || input0_shape.size() == 4)
                << "Currently only support 3D or 4D input";
            int type = generic_op->localOpConfig.getRoot()["type"];
            string expression_template;
            string expression_code;
            if (input0_shape.size() == 3){
                if (type == 0)
                {
                    expression_template =
                        R"(@output0@[B0, N0, N1, N2, N3] = @input0@[B0, (N0 * 16 + N2) // 16 * 16 + (N0 * 16 + N2)  % 8 * 2 + (N1 * 16 + N3) % 16 // 8, (N1 * 16 + N3) // 16 * 16 + (N0 * 16 + N2)  % 16 // 8 * 8 + (N1 * 16 + N3) % 8] where N0 in @N0@, N1 in @N1@, N2 in @N2@, N3 in @N3@;)";
                }
                else if (type == 1)
                {
                    // B[vi // 16, vj // 16, vi % 16, vj % 16] = A[vi // 8 * 8 + vi % 4 * 2 + vj % 16 // 8, vj // 16 * 16 + vi % 8 // 4 * 8 + vj % 8]
                    expression_template =
                        R"(@output0@[B0, N0, N1, N2, N3] = @input0@[B0, (N0 * 16 + N2) // 8 * 8 + (N0 * 16 + N2)  % 4 * 2 + (N1 * 16 + N3) % 16 // 8, (N1 * 16 + N3) // 16 * 16 + (N0 * 16 + N2)  % 8 // 4 * 8 + (N1 * 16 + N3) % 8] where N0 in @N0@, N1 in @N1@, N2 in @N2@, N3 in @N3@;)";
                }
                else
                {
                    NNFUSION_CHECK_FAIL() << "Permutate type not supported";
                }
                nnfusion::json config;
                config["N0"] = input0_shape[1] /
                               static_cast<int>(generic_op->localOpConfig.getRoot()["inner_i"]);
                config["N1"] = input0_shape[2] /
                               static_cast<int>(generic_op->localOpConfig.getRoot()["inner_j"]);
                config["N2"] = generic_op->localOpConfig.getRoot()["inner_i"];
                config["N3"] = generic_op->localOpConfig.getRoot()["inner_j"];
                expression_code = op::create_code_from_template(expression_template, config);
            }
            else if (input0_shape.size() == 4)
            {
                if (type == 0)
                {
                    expression_template =
                        R"(@output0@[B0, B1, N0, N1, N2, N3] = @input0@[B0, B1, (N0 * 16 + N2) // 16 * 16 + (N0 * 16 + N2)  % 8 * 2 + (N1 * 16 + N3) % 16 // 8, (N1 * 16 + N3) // 16 * 16 + (N0 * 16 + N2)  % 16 // 8 * 8 + (N1 * 16 + N3) % 8] where N0 in @N0@, N1 in @N1@, N2 in @N2@, N3 in @N3@;)";
                }
                else if (type == 1)
                {
                    // B[vi // 16, vj // 16, vi % 16, vj % 16] = A[vi // 8 * 8 + vi % 4 * 2 + vj % 16 // 8, vj // 16 * 16 + vi % 8 // 4 * 8 + vj % 8]
                    expression_template =
                        R"(@output0@[B0, B1, N0, N1, N2, N3] = @input0@[B0, B1, (N0 * 16 + N2) // 8 * 8 + (N0 * 16 + N2)  % 4 * 2 + (N1 * 16 + N3) % 16 // 8, (N1 * 16 + N3) // 16 * 16 + (N0 * 16 + N2)  % 8 // 4 * 8 + (N1 * 16 + N3) % 8] where N0 in @N0@, N1 in @N1@, N2 in @N2@, N3 in @N3@;)";
                }
                else
                {
                    NNFUSION_CHECK_FAIL() << "Permutate type not supported";
                }
                nnfusion::json config;
                config["N0"] = input0_shape[2] /
                               static_cast<int>(generic_op->localOpConfig.getRoot()["inner_i"]);
                config["N1"] = input0_shape[3] /
                               static_cast<int>(generic_op->localOpConfig.getRoot()["inner_j"]);
                config["N2"] = generic_op->localOpConfig.getRoot()["inner_i"];
                config["N3"] = generic_op->localOpConfig.getRoot()["inner_j"];
                expression_code = op::create_code_from_template(expression_template, config);
            }

            return expression_code;
        });

REGISTER_OP(LayoutDot)
    .attr<bool>("transpose_A")
    .attr<bool>("transpose_B")
    .attr<size_t>("inner_i")
    .attr<size_t>("inner_j")
    .attr<int>("output_layout")
    .infershape(
        [](std::shared_ptr<graph::GNode> gnode) -> void
        {
            auto generic_op =
                std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
            bool trans_a = generic_op->localOpConfig.getRoot()["transpose_A"];
            bool trans_b = generic_op->localOpConfig.getRoot()["transpose_B"];

            //TODO(leiwang1999):currently only support for NT Layout
            NNFUSION_CHECK(2 == gnode->get_input_size());
            // input 0 shape is B, S, K, input 1 is K, N
            // output sahpe is B, S, N
            auto input0_shape = nnfusion::Shape(gnode->get_input_shape(0));
            auto input1_shape = nnfusion::Shape(gnode->get_input_shape(1));
            NNFUSION_CHECK(input0_shape.size() == 2 || input0_shape.size() == 3 ||
                           input1_shape.size() == 2);
            if (input0_shape.size() == 2)
            {
                nnfusion::Shape output_shape{trans_a ? input0_shape[1]: input0_shape[0],
                                             trans_b ? input1_shape[0]: input1_shape[1] };
                gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
            }
            else if (input0_shape.size() == 3)
            {
                nnfusion::Shape output_shape{input0_shape[0],
                                             trans_a ? input0_shape[2] : input0_shape[1],
                                             trans_b ? input1_shape[0] : input1_shape[1] 
                                            };
                gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
            }
            // print trans_a and trans_b
            NNFUSION_LOG(INFO) << "transa, b is " << trans_a << " " << trans_b;
            // print input0 shape and input1 shape
            NNFUSION_LOG(INFO) << "input0 shape is " << gnode->get_input_shape(0);
            NNFUSION_LOG(INFO) << "input1 shape is " << gnode->get_input_shape(1);
            NNFUSION_LOG(INFO) << "output shape is " << gnode->get_output_shape(0);
        })
    .translate_v2(
        [](std::shared_ptr<graph::GNode> curr) -> std::string
        {
            // todo(leiwang1999): apply correct experession.
            auto generic_op =
                std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
            int output_layout = generic_op->localOpConfig.getRoot()["output_layout"];
            bool trans_a = generic_op->localOpConfig.getRoot()["transpose_A"];
            bool trans_b = generic_op->localOpConfig.getRoot()["transpose_B"];

            string fuse_template =
                R"( temp0@A_fused_layout@ +=! @input0@@A_layout@ where M in @M@;)";
            string compute_template =
                R"( @output0@[M, N] +=! temp0@A_fused_layout@ * @input1@@B_layout@; )";
            string ir_template = fuse_template + compute_template;
            op::OpConfig::any op_config;
            op_config["M"] = 16384;
            op_config["A_fused_layout"] = trans_a? "[K, M]" : "[M, K]";
            op_config["B_layout"] = trans_b? "[N, K]" : "[K, N]";

            auto A_shape = curr->get_input_shape(0);
            int raxis = A_shape.size() - 1;
            string A_layout;
            size_t stride = 16384;
            for (int i = 0; i < A_shape.size(); i++)
            {
                if (i > 0)
                    A_layout += ", ";
                if (i == raxis)
                    A_layout += "K";
                else
                {
                    stride /= A_shape[i];
                    A_layout += "M//" + to_string(stride) + "%" + to_string(A_shape[i]);
                }
            }
            op_config["A_layout"] = "[" + A_layout + "]";

            auto ir = op::create_code_from_template(ir_template, op_config);

            if (curr->get_output_element_type(0) == nnfusion::element::f16)
            {
                ir += "## @: output_layout=" + to_string(output_layout);
            }
            return ir;
        });

REGISTER_OP(LayoutBMM)
    .attr<nnfusion::op::OpConfig::any>("adj_x", {{"b", false}})
    .attr<nnfusion::op::OpConfig::any>("adj_y", {{"b", false}})
    .attr<size_t>("inner_i", 16)
    .attr<size_t>("inner_j", 16)
    .attr<int>("output_layout", 0)
    .constrait(
        [](const nnfusion::op::OpConfig::any& config) -> bool
        {
            if (!config["adj_x"]["b"].is_boolean())
                return false;
            if (!config["adj_y"]["b"].is_boolean())
                return false;
            return true;
        })
    .infershape(
        [](std::shared_ptr<graph::GNode> gnode) -> void
        {
            NNFUSION_CHECK(gnode->get_input_size() == 2);
            const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
            const nnfusion::Shape& input_shape_1 = gnode->get_input_shape(1);
            nnfusion::Shape output_shape_0;

            NNFUSION_CHECK(input_shape_0.size() == input_shape_1.size());
            NNFUSION_CHECK(gnode->get_input_element_type(0) == gnode->get_input_element_type(1));

            for (int i = 0; i < input_shape_0.size() - 2; i++)
            {
                NNFUSION_CHECK(input_shape_0[i] == input_shape_1[i]);
                output_shape_0.push_back(input_shape_0[i]);
            }

            int m0 = input_shape_0[input_shape_0.size() - 2],
                n0 = input_shape_0[input_shape_0.size() - 1];
            int m1 = input_shape_1[input_shape_1.size() - 2],
                n1 = input_shape_1[input_shape_1.size() - 1];

            auto generic_op =
                std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
            bool trans_A = generic_op->localOpConfig.getRoot()["adj_x"]["b"];
            bool trans_B = generic_op->localOpConfig.getRoot()["adj_y"]["b"];

            if (!trans_A && !trans_B)
                NNFUSION_CHECK(m1 == n0), output_shape_0.push_back(m0),
                    output_shape_0.push_back(n1);
            else if (!trans_A && trans_B)
                NNFUSION_CHECK(n0 == n1), output_shape_0.push_back(m0),
                    output_shape_0.push_back(m1);
            else if (trans_A && !trans_B)
                NNFUSION_CHECK(m0 == m1), output_shape_0.push_back(n0),
                    output_shape_0.push_back(n1);
            else // trans_A && trans_B
                NNFUSION_CHECK(m0 == n1), output_shape_0.push_back(n0),
                    output_shape_0.push_back(m1);
            gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
        })
    .translate_v2(
        [](std::shared_ptr<graph::GNode> curr) -> std::string
        {
            NNFUSION_CHECK(curr->get_input_size() == 2);

            const nnfusion::Shape& input_shape_0 = curr->get_input_shape(0);
            const nnfusion::Shape& input_shape_1 = curr->get_input_shape(1);
            nnfusion::Shape output_shape_0 = curr->get_output_shape(0);

            NNFUSION_CHECK(input_shape_0.size() == input_shape_1.size());
            NNFUSION_CHECK(curr->get_input_element_type(0) == curr->get_input_element_type(1));

            auto generic_op =
                std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
            bool trans_A = generic_op->localOpConfig.getRoot()["adj_x"]["b"];
            bool trans_B = generic_op->localOpConfig.getRoot()["adj_y"]["b"];

            auto ir_template =
                R"( @output0@@output0_layout@ +=! @input0@@input0_layout@ * @input1@@input1_layout@; )";

            std::vector<std::string> output0_layout;
            std::vector<std::string> input0_layout;
            std::vector<std::string> input1_layout;

            for (size_t i = 0; i < output_shape_0.size() - 2; ++i)
            {
                std::string batch_dim = "B" + to_string(i);
                output0_layout.push_back(batch_dim);
                input0_layout.push_back(batch_dim);
                input1_layout.push_back(batch_dim);
            }

            output0_layout.push_back("N");
            output0_layout.push_back("M");

            if (trans_A)
            {
                input0_layout.push_back("K");
                input0_layout.push_back("N");
            }
            else
            {
                input0_layout.push_back("N");
                input0_layout.push_back("K");
            }

            if (trans_B)
            {
                input1_layout.push_back("M");
                input1_layout.push_back("K");
            }
            else
            {
                input1_layout.push_back("K");
                input1_layout.push_back("M");
            }

            op::OpConfig::any op_config;
            op_config["input0_layout"] = vector_to_string<std::vector<std::string>>(input0_layout);
            op_config["input1_layout"] = vector_to_string<std::vector<std::string>>(input1_layout);
            op_config["output0_layout"] =
                vector_to_string<std::vector<std::string>>(output0_layout);

            auto ir = op::create_code_from_template(ir_template, op_config);

            int output_layout = generic_op->localOpConfig.getRoot()["output_layout"];
            if (curr->get_output_element_type(0) == nnfusion::element::f16)
            {
                ir += "## @: output_layout=" + to_string(output_layout);
            }
            return ir;
        });

REGISTER_OP(QuantLinear)
    .attr<int>("bits", 4)
    .attr<bool>("transpose_A", false)
    .attr<bool>("transpose_B", true)
    .infershape(
        [](std::shared_ptr<graph::GNode> gnode) -> void
        {
            auto generic_op =
                std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
            bool trans_a = generic_op->localOpConfig.getRoot()["transpose_A"];
            bool trans_b = generic_op->localOpConfig.getRoot()["transpose_B"];
            NNFUSION_CHECK(trans_a == false) << "Currently only support non-transpose A";
            NNFUSION_CHECK(trans_b == true) << "Currently only support transpose B";
            NNFUSION_CHECK(4 == gnode->get_input_size());
            // input 0 shape is B, S, K, input 1 is K, N
            // output sahpe is B, S, N
            auto input_shape = nnfusion::Shape(gnode->get_input_shape(0));
            auto qweight_shape = nnfusion::Shape(gnode->get_input_shape(1));
            NNFUSION_CHECK(input_shape.size() == 2 || input_shape.size() == 3);
            NNFUSION_CHECK(qweight_shape.size() == 2);
            if (input_shape.size() == 2)
            {
                nnfusion::Shape output_shape{trans_a ? input_shape[1] : input_shape[0],
                                             trans_b ? qweight_shape[0] : qweight_shape[1]};
                gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
            }
            else if (input_shape.size() == 3)
            {
                nnfusion::Shape output_shape{input_shape[0],
                                             trans_a ? input_shape[2] : input_shape[1],
                                             trans_b ? qweight_shape[0] : qweight_shape[1]};
                gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
            }

            // print input0 shape and input1 shape
            NNFUSION_LOG(INFO) << "input0 shape is " << gnode->get_input_shape(0);
            NNFUSION_LOG(INFO) << "input1 shape is " << gnode->get_input_shape(1);
            NNFUSION_LOG(INFO) << "output shape is " << gnode->get_output_shape(0);
        })
    .translate_v2(
        [](std::shared_ptr<graph::GNode> curr) -> std::string
        {
            auto _op = static_pointer_cast<nnfusion::op::Dot>(curr->get_op_ptr());
            NNFUSION_CHECK_NOT_NULLPTR(_op)
                << "Node type is not " << curr->get_op_ptr()->get_op_type();
            auto input_shape = curr->get_input_shape(0);
            auto qweight_shape = curr->get_input_shape(1);
            auto scales_shape = curr->get_input_shape(2);
            auto zeros_shape = curr->get_input_shape(3);

            auto ir_template =
                R"( @output0@@output0_layout@ = @input0@@input0_layout@ + @input1@@input1_layout@ + @input2@@input2_layout@ + @input3@@input3_layout@; )";

            vector<string> input_layout, qweight_layout, scales_layout, zeros_layout, output_layout;

            for (size_t i = 0; i + 2 < qweight_shape.size(); i++)
            {
                input_layout.push_back("S" + std::to_string(i));
                output_layout.push_back("S" + std::to_string(i));
            }

            output_layout.push_back("N");
            output_layout.push_back("M");
            input_layout.push_back(_op->get_transpose_A() ? "K" : "N");
            input_layout.push_back(_op->get_transpose_A() ? "N" : "K");
            qweight_layout.push_back(_op->get_transpose_B() ? "M" : "K");
            qweight_layout.push_back(_op->get_transpose_B() ? "K" : "M");
            scales_layout.push_back("M");
            zeros_layout.push_back("M");

            for (size_t i = 0; i + 2 < input_shape.size(); i++)
            {
                qweight_layout.push_back("E" + std::to_string(i));
                output_layout.push_back("E" + std::to_string(i));
            }

            op::OpConfig::any op_config;
            op_config["input0_layout"] = nnfusion::vector_to_string(input_layout);
            op_config["input1_layout"] = nnfusion::vector_to_string(qweight_layout);
            op_config["input2_layout"] = nnfusion::vector_to_string(scales_layout);
            op_config["input3_layout"] = nnfusion::vector_to_string(zeros_layout);
            op_config["output0_layout"] = nnfusion::vector_to_string(output_layout);
            auto ir = op::create_code_from_template(ir_template, op_config);
            return ir;
        });
