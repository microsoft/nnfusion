// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// tool to generate optimized code for a input model with given backend.
// compile and run with:
// g++ ./nnfusion.cpp -std=c++11 -I$HOME/ngraph_dist/include -L$HOME/ngraph_dist/lib -lngraph -o nnfusion
// env LD_LIBRARY_PATH=$HOME/ngraph_dist/lib ./nnfusion

#include <fstream>
#include <iomanip>

#include "gflags/gflags.h"
#include "nnfusion/frontend/onnx_import/onnx.hpp"
#include "nnfusion/frontend/tensorflow_import/tensorflow.hpp"
#include "nnfusion/frontend/torchscript_import/torchscript.hpp"
#include "nnfusion/frontend/util/parameter.hpp"
#include "nnfusion/core/operators/op.hpp"

#include "nnfusion/engine/device/cpu.hpp"
#include "nnfusion/engine/device/cuda.hpp"
#include "nnfusion/engine/device/graphcore.hpp"
#include "nnfusion/engine/device/hlsl.hpp"
#include "nnfusion/engine/device/rocm.hpp"

using namespace std;

DECLARE_bool(fmulti_shape);
DEFINE_string(format,
              "tensorflow",
              "-f, Model file format (tensorflow(default) or torchscript, onnx)");

DECLARE_string(fdefault_device);

DECLARE_string(params);

void display_help()
{
    cout << R"###(
DESCRIPTION
    Generate optimized code for ngraph json model with given backend.

SYNOPSIS
        nnfusion <filename> [--format <format>] [other ...]

OPTIONS
)###";
}

bool file_exists(const string& filename)
{
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

int main(int argc, char** argv)
{
    bool failed = false;
    string model, format, params, backend = "NNFusion";

    model = format = params = "##UNSET##";

    if (argc > 1)
    {
        model = argv[1];
    }
    else
    {
        display_help();
        GFLAGS_NAMESPACE::ShowUsageWithFlags(argv[0]);
        return 1;
    }

    // To support abbreviation along Gflags;
    for (size_t i = 2; i < argc; i++)
    {
        string arg = argv[i];
        if (arg == "-f")
        {
            format = argv[++i];
        }
        else if (arg == "-p")
        {
            params = argv[++i];
        }
    }

    google::SetUsageMessage(argv[0]);
    google::AllowCommandLineReparsing();
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (format == "##UNSET##")
        format = FLAGS_format;

    if (params == "##UNSET##")
        params = FLAGS_params;

    if (!model.empty() && !file_exists(model))
    {
        cout << "File " << model << " not found\n";
        failed = true;
    }

    if (failed)
    {
        display_help();
        return 1;
    }

    cout << "\n";
    cout << "============================================================================\n";
    cout << "---- Processing '" << model << "'\n";
    cout << "============================================================================\n";
    // try
    // {
    shared_ptr<nnfusion::graph::Graph> graph = nullptr;
    std::vector<shared_ptr<nnfusion::graph::Graph>> vec_graph;
    if (format == "tensorflow")
    {
        // load tensorflow model as graph
        graph = nnfusion::frontend::load_tensorflow_model(model);
    }
#if TORCHSCRIPT_FRONTEND
    else if (format == "torchscript")
    {
        std::vector<nnfusion::frontend::ParamInfo> params_vec;
        if (params != "##UNSET##")
        {
            params_vec = nnfusion::frontend::build_torchscript_params_from_string(params);
        }
        graph = nnfusion::frontend::load_torchscript_model(model, params_vec);
    }
#endif
#if ONNX_FRONTEND
    else if (format == "onnx")
    {
        if (FLAGS_fmulti_shape)
        {
            auto vec_dim_params = nnfusion::frontend::build_multi_onnx_params_from_string(params);
            for (auto& dim_params : vec_dim_params)
            {
                GNode::reset_next_instance_id();
                op::Op::reset_next_instance_id();
                descriptor::Tensor::reset_next_instance_id();
                vec_graph.push_back(nnfusion::frontend::load_onnx_model(model, dim_params));
            }
        }
        else
        {
            std::unordered_map<std::string, SymDim> dim_params;
            if (params != "##UNSET##")
            {
                dim_params = nnfusion::frontend::build_onnx_params_from_string(params);
            }
            graph = nnfusion::frontend::load_onnx_model(model, dim_params);
        }
    }
#endif
    else
    {
        throw nnfusion::errors::NotSupported("Unsupported model format '" + format +
                                             "' in NNFusion");
    }

    if (!backend.empty() && graph != nullptr)
    {
        if (!FLAGS_fdefault_device.empty())
        {
            // auto runtime = ngraph::runtime::Backend::create(backend);
            nnfusion::engine::CudaEngine cuda_engine;
            nnfusion::engine::HLSLEngine hlsl_engine;
            nnfusion::engine::GraphCoreEngine gc_engine;
            nnfusion::engine::ROCmEngine rocm_engine;
            nnfusion::engine::CpuEngine cpu_engine;

            switch (get_device_type(FLAGS_fdefault_device))
            {
            case CUDA_GPU:
                cuda_engine.run_on_graph(graph);
                break;
            // case CUDA_GPU:
            //     runtime->codegen(graph);
            //     break;
            case ROCM_GPU:
                rocm_engine.run_on_graph(graph);
                break;
            // case ROCM_GPU: runtime->codegen(graph); break;
            // case GENERIC_CPU: runtime->codegen(graph); break;
            case GENERIC_CPU: cpu_engine.run_on_graph(graph); break;
            case HLSL: hlsl_engine.run_on_graph(graph); break;
            case GraphCore: gc_engine.run_on_graph(graph); break;
            default:
                throw nnfusion::errors::NotSupported("Unsupported device type:" +
                                                     FLAGS_fdefault_device);
            }
        }
        else
        {
            throw nnfusion::errors::InvalidArgument("Default device cannot be empty.");
        }
    }

    // Merged graph needs to be:
    // 1. same graph structure;
    // 2. different shape;
    // 3. constant memory must has same layout
    // ./build/src/tools/nnfusion/nnfusion  ./test/models/onnx/abs.onnx -f onnx -multi_shape=true -p "{seq:1500;past_seq:0}, {seq:1;past_seq:2048}"
    if (!backend.empty() && FLAGS_fmulti_shape && !vec_graph.empty())
    {
        NNFUSION_LOG(INFO) << "Graph count: " << vec_graph.size() << "\n";
        nnfusion::engine::CudaMultiEngine cuda_multi_engine;
        nnfusion::engine::HLSLMultiEngine hlsl_multi_engine;
        switch (get_device_type(FLAGS_fdefault_device))
        {
            case CUDA_GPU: cuda_multi_engine.run_on_graphs(vec_graph); break;
            case HLSL: hlsl_multi_engine.run_on_graphs(vec_graph); break;
        }
    }
    return 0;
}
