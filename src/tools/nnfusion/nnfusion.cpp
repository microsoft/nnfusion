// tool to generate optimized code for a input model with given backend.
// compile and run with:
// g++ ./nnfusion.cpp -std=c++11 -I$HOME/ngraph_dist/include -L$HOME/ngraph_dist/lib -lngraph -o nnfusion
// env LD_LIBRARY_PATH=$HOME/ngraph_dist/lib ./nnfusion

#include <fstream>
#include <iomanip>

#include "gflags/gflags.h"
#include "nnfusion/engine/external/backend.hpp"
#include "nnfusion/frontend/tensorflow_import/tensorflow.hpp"

using namespace std;

DEFINE_string(format, "tensorflow", "-f, Model file format (tensorflow(default) or onnx)");

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
    string model, format, backend = "NNFusion";

    model = format = "##UNSET##";

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
    }

    google::SetUsageMessage(argv[0]);
    google::AllowCommandLineReparsing();
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (format == "##UNSET##")
        format = FLAGS_format;

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
    try
    {
        shared_ptr<nnfusion::graph::Graph> graph = nullptr;
        if (format == "tensorflow")
        {
            // load tensorlfow model as graph
            graph = nnfusion::frontend::load_tensorflow_model(model);
        }
        else if (format == "onnx")
        {
            //graph = ngraph::onnx_import::import_onnx_function(model);
        }
        else
        {
            throw nnfusion::errors::NotSupported("Unsupported model format '" + format +
                                                 "' in NNFusion");
        }

        if (!backend.empty())
        {
            auto runtime = ngraph::runtime::Backend::create(backend);
            runtime->codegen(graph);
        }
    }
    catch (exception& e)
    {
        cout << "Exception caught on '" << model << "'\n" << e.what() << endl;
    }

    return 0;
}
