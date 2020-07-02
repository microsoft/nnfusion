//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <fstream>

#include "tensorflow.hpp"

namespace nnfusion
{
    namespace frontend
    {
        std::shared_ptr<nnfusion::graph::Graph> load_tensorflow_model(std::istream& sin)
        {
            tensorflow::GraphDef tensorflow_graph;
            CHECK(tensorflow_graph.ParseFromIstream(&sin))
                << "failure parsing data from the stream";

            LOG(INFO) << "Import Tensorflow Graph Size: [" << tensorflow_graph.ByteSizeLong()
                      << "]";

            auto graph_convert = tensorflow_import::GraphConvert{tensorflow_graph};

            std::shared_ptr<nnfusion::graph::Graph> graph = graph_convert.get_graph();
            return graph;
        }

        std::shared_ptr<nnfusion::graph::Graph> load_tensorflow_model(const std::string& path)
        {
            std::ifstream ifs{path, std::ios::in | std::ios::binary};
            CHECK(ifs.is_open()) << "failure opening file:" + path;
            return load_tensorflow_model(ifs);
        }

        // void register_operator(const std::string& name,
        //                        std::int64_t version,
        //                        const std::string& domain,
        //                        Operator fn)
        // {
        //     OperatorsBridge::register_operator(name, version, domain, std::move(fn));
        // }

    } // namespace frontend

} // namespace nnfusion
