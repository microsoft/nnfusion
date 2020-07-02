// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <sstream>

#include "backend.hpp"
#include "backend_manager.hpp"

using namespace std;
using namespace ngraph;

runtime::Backend::~Backend()
{
}

unique_ptr<runtime::Backend> runtime::Backend::create(const string& type)
{
    return BackendManager::create_backend(type);
}

vector<string> runtime::Backend::get_registered_devices()
{
    return BackendManager::get_registered_backends();
}

DEFINE_int32(min_log_level,
             0,
             "Minimum logging level: 0 - debug; 1 - info; 2 - warning; 3 - error; 4 - fatal;");

extern "C" const char* get_ngraph_version_string()
{
    return "nnfusion_engine";
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    runtime::Backend* backend = nullptr;
    string type(configuration_string);
    backend = new cuda_codegen();
    return backend;
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
}

cuda_codegen::cuda_codegen()
    : nnfusion_Backend()
    , m_functrans(new Interpreter)
{
}

bool cuda_codegen::codegen(shared_ptr<graph::Graph> graph)
{
    TranslationUnit& graph_unit = m_graph_map[graph];
    if (graph_unit.m_is_translated == false)
    {
        auto tus = m_functrans->translate(graph);
        CHECK_NOT_NULLPTR(tus);
    }
    return true;
}