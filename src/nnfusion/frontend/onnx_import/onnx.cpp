//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <fstream>

#include "onnx.hpp"
#include "util/graph_convert.hpp"

DEFINE_string(params,
              "##UNSET##",
              "-p, Model input shape and type, fot torchscript, it's full shape like "
              "\"1,1:float;2,3,4,5:double\", for onnx, it's dynamic dim like "
              "\"dim1_name:4;dim2_name:128\"");

DEFINE_bool(fort_folding, true, "use onnxruntime to do constant folding.");
DEFINE_bool(fincrease_precision, false, "Increase precision for fp16 model");
namespace nnfusion
{
    namespace frontend
    {
        std::shared_ptr<nnfusion::graph::Graph>
            load_onnx_model(std::istream& sin,
                            const std::string& model_dir,
                            const std::unordered_map<std::string, SymDim>& dim_params)
        {
            onnx::ModelProto onnx_graph;
            NNFUSION_CHECK(onnx_graph.ParseFromIstream(&sin))
                << "failure parsing data from the stream";

            NNFUSION_LOG(INFO) << "Import ONNX Graph Size: [" << onnx_graph.ByteSizeLong() << "]";
            // TODO: this is a hardcode for BERT training
            // std::map<std::string, size_t> dim_map = {
            //     {"batch", 2}, {"sequence", 512}, {"dynamic_prediction_count", 20}};
            auto graph_convert = onnx_import::GraphConvert{onnx_graph, dim_params, model_dir};

            std::shared_ptr<nnfusion::graph::Graph> graph = graph_convert.get_graph();
            return graph;
        }

        std::shared_ptr<nnfusion::graph::Graph>
            load_onnx_model(const std::string& path,
                            const std::unordered_map<std::string, SymDim>& dim_params)
        {
            string m_path = path;
            std::ifstream opt_fin;
            string mp_filename = string(nnfusion::tmpnam(nullptr));
            if (FLAGS_fincrease_precision)
            {
                NNFUSION_LOG(INFO) << "Increase ONNX Model Precision with External Tool "
                                      "(models/pytorch2onnx/increase_precision.py)";
                string script_path =
                    nnfusion::codegen::get_file_from_templates("onnx/increase_precision.py");
                string cmd = "python3 " + script_path +
                             " --file " +
                             m_path + " --mp_file " + mp_filename;
                
                int sys_ret = system(cmd.c_str());
                // NNFUSION_LOG(INFO) << "mix precision model path: " << mp_filename;
                opt_fin = std::ifstream(mp_filename.c_str());
                if (sys_ret == 0 && opt_fin.is_open())
                {
                    m_path = mp_filename;
                }
                else
                {
                    NNFUSION_LOG(NNFUSION_WARNING)
                        << "Failed to increase ONNX Graph  precision with external tool, please "
                           "check error messages reported by the tool, fallback";
                }
            }
           
            string optimized_filename = string(nnfusion::tmpnam(nullptr));
            if (FLAGS_fort_folding)
            {
                NNFUSION_LOG(INFO) << "Optimizing ONNX Graph with External Tool "
                                      "(models/pytorch2onnx/ort_run_frozen.py)";
                string script_path =
                    nnfusion::codegen::get_file_from_templates("onnx/ort_run_frozen.py");
                string cmd = "python3 " + script_path +
                             " --graph_optimization_level ORT_ENABLE_BASIC "
                             "--warmup 1 --iters 0 --file " +
                             m_path + " --optimized_model_filepath " + optimized_filename;
                if (dim_params.size() > 0)
                {
                    string dim_params_str = " --symbolic_dims \'{";
                    for (auto& it : dim_params)
                    {
                        if (dim_params_str != " --symbolic_dims \'{")
                        {
                            dim_params_str += ", ";
                        }
                        dim_params_str += "\"" + it.first + "\": " + to_string(it.second.max());
                    }
                    dim_params_str += "}\'";
                    cmd += dim_params_str;
                }
                int sys_ret = system(cmd.c_str());
                opt_fin = std::ifstream(optimized_filename.c_str());
                if (sys_ret == 0 && opt_fin.is_open())
                {
                    m_path = optimized_filename;
                }
                else
                {
                    NNFUSION_LOG(NNFUSION_WARNING)
                        << "Failed to optimize ONNX Graph with external tool, please "
                           "check error messages reported by the tool, fallback";
                }
            }
            std::ifstream ifs{m_path, std::ios::in | std::ios::binary};
            NNFUSION_CHECK(ifs.is_open()) << "failure opening file:" + path;
            string model_dir = "";
            string weight_path = FLAGS_fincrease_precision ? m_path : path;
            auto pos = weight_path.rfind("/");
            if (pos != std::string::npos)
            {
                model_dir = weight_path.substr(0, pos);
            }

            auto graph = load_onnx_model(ifs, model_dir, dim_params);

            if (opt_fin.is_open())
            {
                remove(optimized_filename.c_str());
            }

            return graph;
        }
    } // namespace frontend
} // namespace nnfusion
