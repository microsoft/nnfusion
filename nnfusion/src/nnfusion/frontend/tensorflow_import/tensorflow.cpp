//  Copyright (c) Microsoft Corporation.
//  Licensed under the MIT License.

#include <fstream>

#include "tensorflow.hpp"

namespace nnfusion
{
    namespace frontend
    {
        std::shared_ptr<nnfusion::graph::Graph> load_tensorflow_model(std::istream& sin)
        {
            tensorflow::GraphDef tensorflow_graph;
            NNFUSION_CHECK(tensorflow_graph.ParseFromIstream(&sin))
                << "failure parsing data from the stream";

            NNFUSION_LOG(INFO) << "Import Tensorflow Graph Size: ["
                               << tensorflow_graph.ByteSizeLong() << "]";

            auto graph_convert = tensorflow_import::GraphConvert{tensorflow_graph};

            std::shared_ptr<nnfusion::graph::Graph> graph = graph_convert.get_graph();
            return graph;
        }

        std::shared_ptr<nnfusion::graph::Graph> load_tensorflow_model(const std::string& path)
        {
            std::ifstream ifs{path, std::ios::in | std::ios::binary};
            NNFUSION_CHECK(ifs.is_open()) << "failure opening file:" + path;
            return load_tensorflow_model(ifs);
        }
    } // namespace frontend
} // namespace nnfusion
