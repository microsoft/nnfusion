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
