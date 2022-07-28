// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "contrib/custom_op/custom_op.h"

REGISTER_OP(GridSample)
    .attr<std::string>("mode", "bilinear")
    .attr<bool>("align_corners", false)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(2 == gnode->get_input_size());
        auto& input_shape = gnode->get_input_shape(0);
        auto& grid_shape = gnode->get_input_shape(1);
        Shape output_shape(input_shape);
        output_shape[2] = grid_shape[1];
        output_shape[3] = grid_shape[2];
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
    })
    .antares_ir([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto& input_shape = gnode->get_input_shape(0);
        auto height = input_shape[2];
        auto width = input_shape[3];
        
        std::vector<std::string> seq_cmds;
        // Un-normed Y-X coordinates
        // seq_cmds.push_back("mediate0[N, H, W] = (@input1@[N, H, W, 1] + 1) / 2 * (@height@ - 1);");
        // seq_cmds.push_back("mediate1[N, H, W] = (@input1@[N, H, W, 0] + 1) / 2 * (@width@ - 1);");
        seq_cmds.push_back("mediate0[N, H, W] = ((@input1@[N, H, W, 1] + 1) * @height@ - 1) / 2;");
        seq_cmds.push_back("mediate1[N, H, W] = ((@input1@[N, H, W, 0] + 1) * @width@ - 1) / 2;");
        
        // Corners coordinates
        seq_cmds.push_back("mediate2[N, H, W] = mediate0[N, H, W].cast(`int32`);");
        seq_cmds.push_back("mediate3[N, H, W] = mediate1[N, H, W].cast(`int32`);");
        seq_cmds.push_back("mediate4[N, H, W] = mediate0[N, H, W].cast(`int32`) + 1;");
        seq_cmds.push_back("mediate5[N, H, W] = mediate1[N, H, W].cast(`int32`) + 1;");

        // Left-top corner
        seq_cmds.push_back("mediate6[N, H, W] = (mediate4[N, H, W].cast(`float32`) - mediate0[N, H, W]) * (mediate5[N, H, W].cast(`float32`) - mediate1[N, H, W]);");
        seq_cmds.push_back("mediate7[N, C, H, W] = @input0@[N, C, mediate2[N, H, W], mediate3[N, H, W]].when([mediate2[N, H, W] >= 0, mediate3[N, H, W] >= 0, mediate2[N, H, W] < @height@, mediate3[N, H, W] < @width@], const(0.0).cast(`float32`)) * mediate6[N, H, W];");

        // Right-top corner
        seq_cmds.push_back("mediate8[N, H, W] = (mediate4[N, H, W].cast(`float32`) - mediate0[N, H, W]) * (mediate1[N, H, W] - mediate3[N, H, W].cast(`float32`));");
        seq_cmds.push_back("mediate9[N, C, H, W] = @input0@[N, C, mediate2[N, H, W], mediate5[N, H, W]].when([mediate2[N, H, W] >= 0, mediate5[N, H, W] >= 0, mediate2[N, H, W] < @height@, mediate5[N, H, W] < @width@], const(0.0).cast(`float32`)) * mediate8[N, H, W];");

        // Left-bottom corner
        seq_cmds.push_back("mediate10[N, H, W] = (mediate0[N, H, W] - mediate2[N, H, W].cast(`float32`)) * (mediate5[N, H, W].cast(`float32`) - mediate1[N, H, W]);");
        seq_cmds.push_back("mediate11[N, C, H, W] = @input0@[N, C, mediate4[N, H, W], mediate3[N, H, W]].when([mediate4[N, H, W] >= 0, mediate3[N, H, W] >= 0, mediate4[N, H, W] < @height@, mediate3[N, H, W] < @width@], const(0.0).cast(`float32`)) * mediate10[N, H, W];");
        
        // Right-top corner
        seq_cmds.push_back("mediate12[N, H, W] = (mediate0[N, H, W] - mediate2[N, H, W].cast(`float32`)) * (mediate1[N, H, W] - mediate3[N, H, W].cast(`float32`));");
        seq_cmds.push_back("mediate13[N, C, H, W] = @input0@[N, C, mediate4[N, H, W], mediate5[N, H, W]].when([mediate4[N, H, W] >= 0, mediate5[N, H, W] >= 0, mediate4[N, H, W] < @height@, mediate5[N, H, W] < @width@], const(0.0).cast(`float32`)) * mediate12[N, H, W];");

        seq_cmds.push_back("@output0@[N, C, H, W] = mediate7[N, C, H, W] + mediate9[N, C, H, W] + mediate11[N, C, H, W] + mediate13[N, C, H, W];");

        auto sep = std::string(" ");
        std::string expr = join<std::vector<std::string>>(seq_cmds, sep);
        expr = op::create_code_from_template(expr, {{"height", to_string(height)},
                                                    {"width", to_string(width)}});
        return expr;
    });
