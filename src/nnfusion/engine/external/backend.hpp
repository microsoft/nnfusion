//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
// Microsoft (c) 2019, Wenxiang Hu

#pragma once

#include "nnfusion/common/shape.hpp"
#include "nnfusion/common/type/element_type.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/engine/interpreter.hpp"

namespace ngraph
{
    namespace runtime
    {
        class ExternalFunction;
        class Tensor;
        class Backend;
    }
}

/// \brief Interface to a generic backend.
///
/// Backends are responsible for function execution and value allocation.
class ngraph::runtime::Backend
{
public:
    virtual ~Backend();
    /// \brief Create a new Backend object
    /// \param type The name of a registered backend, such as "CPU" or "GPU".
    ///   To select a subdevice use "GPU:N" where s`N` is the subdevice number.
    /// \returns unique_ptr to a new Backend or nullptr if the named backend
    ///   does not exist.
    static std::unique_ptr<Backend> create(const std::string& type);

    /// \brief Query the list of registered devices
    /// \returns A vector of all registered devices.
    static std::vector<std::string> get_registered_devices();

    virtual bool codegen(std::shared_ptr<nnfusion::graph::Graph> graph) { return false; }
};

using Backend = ngraph::runtime::Backend;
namespace nnfusion
{
    // This is an abstract class for NNFusion Backend
    class nnfusion_Backend : public Backend
    {
    public:
        nnfusion_Backend()
            : Backend(){};
        virtual bool codegen(std::shared_ptr<graph::Graph> graph) = 0;
    };

    class cuda_codegen : public nnfusion_Backend
    {
    public:
        cuda_codegen();
        bool codegen(shared_ptr<graph::Graph> graph);

    private:
        map<shared_ptr<graph::Graph>, TranslationUnit> m_graph_map;

    protected:
        shared_ptr<Interpreter> m_functrans;
    };
}