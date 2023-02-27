// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Resize)
    .attr<std::string>("method")
    .attr<std::string>("coordinate_transformation_mode")
    .attr<std::string>("nearest_mode")
    .attr<bool>("no_scale")
    .constrait([](const nnfusion::op::OpConfig::any& config) -> bool {
        // Currently support Nearst and Linear mode
        // in Linear mode, the size of output can be specified by sizes value
        if (config["method"] != "NEAREST" && config["method"] != "LINEAR")
        {
            return false;
        }
        // This is specific for SR model
        // The "linear" mode includes linear interpolation for 1D tensor and N-linear interpolation for N-D tensor (for example, bilinear interpolation for 2D tensor).
        //if(config["method"] == "LINEAR" && (config["coordinate_transformation_mode"] != "pytorch_half_pixel" || config["nearest_mode"] != "floor"))
        //{
        //    return false;
        //}
        return true;
    })
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto input_shape = gnode->get_input_shape(0);
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        auto mode = generic_op->localOpConfig.getRoot()["method"];
        auto no_scale = generic_op->localOpConfig.getRoot()["no_scale"];

        if (mode == "NEAREST")
        {
            auto ng_op = gnode->get_in_edge(1)->get_src();
            NNFUSION_CHECK(2 == gnode->get_input_size());
            NNFUSION_CHECK(ng_op->is_constant())
                << "We only accept the Resize input \"scales\" as Constant";
            auto scales = std::dynamic_pointer_cast<nnfusion::op::Constant>(ng_op->get_op_ptr())
                              ->get_vector<float>();
            NNFUSION_CHECK(input_shape.size() == scales.size());

            nnfusion::Shape output_shape(scales.size());
            for (int i = 0; i < scales.size(); i++)
                output_shape[i] = input_shape[i] * scales[i];
            gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
        }
        else if (mode == "LINEAR")
        {
            auto ng_op = gnode->get_in_edge(1)->get_src();
            NNFUSION_CHECK(ng_op->is_constant())
                << "We only accept the Resize input \"scales\" as Constant";
            if (no_scale)
            {
                auto ng_op = gnode->get_in_edge(1)->get_src();
                NNFUSION_LOG(INFO)
                    << std::dynamic_pointer_cast<nnfusion::op::Constant>(ng_op->get_op_ptr())
                           ->get_data_size();
                auto sizes = std::dynamic_pointer_cast<nnfusion::op::Constant>(ng_op->get_op_ptr())
                                 ->get_vector<int64_t>();

                NNFUSION_CHECK(4 == sizes.size());
                nnfusion::Shape output_shape(sizes.size());
                for (int i = 0; i < sizes.size(); i++)
                    output_shape[i] = sizes[i];
                gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
            }
            // don't have scale, only have sizes
            else
            {
                auto scales = std::dynamic_pointer_cast<nnfusion::op::Constant>(ng_op->get_op_ptr())
                                  ->get_vector<float>();
                NNFUSION_CHECK(4 == scales.size());
                nnfusion::Shape output_shape(scales.size());
                for (int i = 0; i < scales.size(); i++)
                    output_shape[i] = input_shape[i] * scales[i];
                gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
            }
        }
    })
    .translate_v2([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        auto mode = generic_op->localOpConfig.getRoot()["method"];
        auto no_scale = generic_op->localOpConfig.getRoot()["no_scale"];
        string dtype;
        NNFUSION_CHECK(
            element::Type::nnfusion_element_type_to_dtype_string(gnode->get_element_type(), dtype));
        if (mode == "NEAREST")
        {
            auto expression_template =
                R"( @output0@@output0_layout@ = @input0@@input0_layout@ where @cond@; )";

            nnfusion::Shape output_shape = gnode->get_output_shape(0);

            auto ng_op = gnode->get_in_edge(1)->get_src();
            NNFUSION_CHECK(ng_op->is_constant())
                << "We only accept the Tile input \"scales\" as Constant.";
            auto scales = std::dynamic_pointer_cast<nnfusion::op::Constant>(ng_op->get_op_ptr())
                              ->get_vector<float>();

            auto output_layout = op::create_layout_from_dims(output_shape);
            auto input_layout = op::create_layout_from_dims(output_shape);
            std::string cond;
            for (int d = 0; d < input_layout.size(); ++d)
            {
                if (scales[d] > 1)
                    input_layout[d] = input_layout[d] + " // " + to_string(int(scales[d]));
                else
                    input_layout[d] = input_layout[d] + " * " + to_string(int(1 / scales[d]));
                cond += (cond.empty() ? "" : ", ") + output_layout[d] + " in " +
                        to_string(output_shape[d]);
            }

            auto expr =
                op::create_code_from_template(expression_template,
                                              {{"output0_layout", vector_to_string(output_layout)},
                                               {"input0_layout", vector_to_string(input_layout)},
                                               {"cond", cond}});
            return expr;
        }
        else if (mode == "LINEAR" &&
                 generic_op->localOpConfig.getRoot()["coordinate_transformation_mode"] !=
                     "align_corners")
        {
            // Only support 2D case
            nnfusion::Shape input_shape = gnode->get_input_shape(0);
            nnfusion::Shape output_shape = gnode->get_output_shape(0);
            auto output_layout = op::create_layout_from_dims(output_shape);

            auto input00_layout = op::create_layout_from_dims(output_shape);
            auto input01_layout = op::create_layout_from_dims(output_shape);
            auto input10_layout = op::create_layout_from_dims(output_shape);
            auto input11_layout = op::create_layout_from_dims(output_shape);

            vector<float> scales;
            auto ng_op = gnode->get_in_edge(1)->get_src();
            if (no_scale)
            {
                for (int i = 0; i < output_shape.size(); i++)
                    scales.push_back((float)output_shape[i] / (float)input_shape[i]);
            }
            else
            {
                scales = std::dynamic_pointer_cast<nnfusion::op::Constant>(ng_op->get_op_ptr())
                             ->get_vector<float>();
            }

            auto expression_template =
                "h_map[CH] = ((CH + 0.5)//@h_scale@ - 0.5).call(`min`, "
                "[const(@h_shape@).cast(`float`)]).call(`max`, [const(0.0)]) where CH in "
                "@oh_shape@;"
                "h_weight[CH] = h_map[CH].call(`remainder`) where CH in "
                "@oh_shape@;"
                "w_map[CH] = ((CH + 0.5)//@w_scale@ - 0.5).call(`min`, "
                "[const(@w_shape@).cast(`float`)]).call(`max`, [const(0.0)]) where CH in "
                "@ow_shape@;"
                "w_weight[CH] = w_map[CH].call(`remainder`) where CH in "
                "@ow_shape@;"
                "@output0@@output0_layout@ = (@input0@@input00_layout@ * (1.0 - "
                "h_weight@h_layout@) "
                "* "
                "(1.0 - w_weight@w_layout@)"
                "+ @input0@@input10_layout@ * (h_weight@h_layout@) * (1.0 - w_weight@w_layout@)"
                "+ @input0@@input01_layout@ * (1.0 - h_weight@h_layout@) * (w_weight@w_layout@)"
                "+ @input0@@input11_layout@ * (h_weight@h_layout@) * ("
                "w_weight@w_layout@)"
                ").cast(`@dtype@`) where @con@;";
            ;

            std::string cond;
            for (int d = 0; d < output_layout.size(); ++d)
                cond += (cond.empty() ? "" : ", ") + output_layout[d] + " in " +
                        to_string(output_shape[d]);

            for (int d = 0; d < 2; ++d)
                input00_layout[d] = input01_layout[d] = input11_layout[d] = input10_layout[d];

            input00_layout[2] = "h_map[" + output_layout[2] + "].cast(`int32`)";
            input00_layout[3] = "w_map[" + output_layout[3] + "].cast(`int32`)";
            input10_layout[2] = "h_map[" + output_layout[2] + "].call(`ceil`)";
            input10_layout[3] = "w_map[" + output_layout[3] + "].cast(`int32`)";
            input01_layout[2] = "h_map[" + output_layout[2] + "].cast(`int32`)";
            input01_layout[3] = "w_map[" + output_layout[3] + "].call(`ceil`)";
            input11_layout[2] = "h_map[" + output_layout[2] + "].call(`ceil`)";
            input11_layout[3] = "w_map[" + output_layout[3] + "].call(`ceil`)";

            vector<std::string> w_layout;
            w_layout.push_back(output_layout[3]);
            vector<std::string> h_layout;
            h_layout.push_back(output_layout[2]);

            auto expr =
                op::create_code_from_template(expression_template,
                                              {{"output0_layout", vector_to_string(output_layout)},
                                               {"h_scale", scales[2]},
                                               {"oh_shape", output_shape[2]},
                                               {"oh_shape_plus_one", output_shape[2] + 1},
                                               {"h_shape", input_shape[2] - 1},
                                               {"w_scale", scales[3]},
                                               {"ow_shape", output_shape[3]},
                                               {"w_shape", input_shape[3] - 1},
                                               {"ow_shape_plus_one", output_shape[3] + 1},
                                               {"input00_layout", vector_to_string(input00_layout)},
                                               {"input01_layout", vector_to_string(input01_layout)},
                                               {"input10_layout", vector_to_string(input10_layout)},
                                               {"input11_layout", vector_to_string(input11_layout)},
                                               {"w_layout", vector_to_string(w_layout)},
                                               {"h_layout", vector_to_string(h_layout)},
                                               {"dtype", dtype},
                                               {"con", cond}});
            return expr;
        }
        else if (mode == "LINEAR" &&
                 generic_op->localOpConfig.getRoot()["coordinate_transformation_mode"] ==
                     "align_corners")
        {
            // Only support 2D case
            nnfusion::Shape input_shape = gnode->get_input_shape(0);
            nnfusion::Shape output_shape = gnode->get_output_shape(0);
            auto output_layout = op::create_layout_from_dims(output_shape);

            auto input00_layout = op::create_layout_from_dims(output_shape);
            auto input01_layout = op::create_layout_from_dims(output_shape);
            auto input10_layout = op::create_layout_from_dims(output_shape);
            auto input11_layout = op::create_layout_from_dims(output_shape);

            vector<float> scales;
            auto ng_op = gnode->get_in_edge(1)->get_src();
            if (no_scale)
            {
                for (int i = 0; i < output_shape.size(); i++)
                    scales.push_back((float)output_shape[i] / (float)input_shape[i]);
            }
            else
            {
                scales = std::dynamic_pointer_cast<nnfusion::op::Constant>(ng_op->get_op_ptr())
                             ->get_vector<float>();
            }

            // Strong assumption that illegal access will not crash!!!!
            auto expression_template =
                "h_map[CH] = (CH * @h_scale@) where CH in @oh_shape@;"
                "w_map[CH] = (CH * @w_scale@) where CH in @ow_shape@;"
                "@output0@@output0_layout@ = ((1.0 - h_map[@N3@].call(`remainder`)) * (1.0 - "
                "w_map[@N4@].call(`remainder`)) * @input0@@input00_layout@ +"
                "(h_map[@N3@].call(`remainder`)) * (1.0 - w_map[@N4@].call(`remainder`)) * "
                "@input0@@input10_layout@ +"
                "(1.0 - h_map[@N3@].call(`remainder`)) * (w_map[@N4@].call(`remainder`)) * "
                "@input0@@input01_layout@ +"
                "(h_map[@N3@].call(`remainder`)) * (w_map[@N4@].call(`remainder`)) * "
                "@input0@@input11_layout@).cast(`@dtype@`) where @con@;";

            std::string cond;
            for (int d = 0; d < output_layout.size(); ++d)
                cond += (cond.empty() ? "" : ", ") + output_layout[d] + " in " +
                        to_string(output_shape[d]);

            for (int d = 0; d < 2; ++d)
                input00_layout[d] = input01_layout[d] = input11_layout[d] = input10_layout[d];

            input00_layout[2] = "h_map[" + output_layout[2] + "].cast(`int32`)";
            input00_layout[3] = "w_map[" + output_layout[3] + "].cast(`int32`)";
            input10_layout[2] = "h_map[" + output_layout[2] + "].call(`ceil`)";
            input10_layout[3] = "w_map[" + output_layout[3] + "].cast(`int32`)";
            input01_layout[2] = "h_map[" + output_layout[2] + "].cast(`int32`)";
            input01_layout[3] = "w_map[" + output_layout[3] + "].call(`ceil`)";
            input11_layout[2] = "h_map[" + output_layout[2] + "].call(`ceil`)";
            input11_layout[3] = "w_map[" + output_layout[3] + "].call(`ceil`)";

            vector<std::string> w_layout;
            w_layout.push_back(output_layout[3]);
            vector<std::string> h_layout;
            h_layout.push_back(output_layout[2]);

            auto expr = op::create_code_from_template(
                expression_template,
                {{"output0_layout", vector_to_string(output_layout)},
                 {"h_scale", (input_shape[2] - 1.0) / (float)(output_shape[2] - 1.0)},
                 {"oh_shape", output_shape[2]},
                 {"oh_shape_plus_one", output_shape[2] + 1},
                 {"h_shape", input_shape[2] - 1},
                 {"w_scale", (input_shape[3] - 1.0) / (float)(output_shape[3] - 1.0)},
                 {"ow_shape", output_shape[3]},
                 {"w_shape", input_shape[3] - 1},
                 {"ow_shape_plus_one", output_shape[3] + 1},
                 {"input00_layout", vector_to_string(input00_layout)},
                 {"input01_layout", vector_to_string(input01_layout)},
                 {"input10_layout", vector_to_string(input10_layout)},
                 {"input11_layout", vector_to_string(input11_layout)},
                 {"w_layout", vector_to_string(w_layout)},
                 {"h_layout", vector_to_string(h_layout)},
                 {"N3", output_layout[2]},
                 {"N4", output_layout[3]},
                 {"dtype", dtype},
                 {"con", cond}});
            return expr;
        }
        return "";
    });
