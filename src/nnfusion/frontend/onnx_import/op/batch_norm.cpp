// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>

#include "ngraph/node_vector.hpp"
#include "nnfusion/core/operators/batch_norm.hpp"

#include "../exceptions.hpp"
#include "../op/batch_norm.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector batch_norm(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    auto x = inputs.at(0);
                    auto scale = inputs.at(1);
                    auto bias = inputs.at(2);
                    std::shared_ptr<ngraph::Node> mean{nullptr};
                    std::shared_ptr<ngraph::Node> var{nullptr};

                    std::int64_t is_test{node.get_attribute_value<std::int64_t>("is_test", 1)};
                    std::int64_t spatial{node.get_attribute_value<std::int64_t>("spatial", 1)};
                    double epsilon{node.get_attribute_value<double>("epsilon", 1e-5)};

                    // TODO: Implement learning mode support
                    // float momentum{node.get_attribute_value<float>("momentum", 0.9f)};
                    ASSERT_IS_SUPPORTED(node, is_test) << "only 'is_test' mode is supported.";
                    ASSERT_IS_SUPPORTED(node, spatial) << "only 'spatial' mode is supported.";

                    if (inputs.size() >= 5)
                    {
                        mean = inputs.at(3);
                        var = inputs.at(4);
                        return {std::make_shared<ngraph::op::BatchNormInference>(
                            epsilon, scale, bias, x, mean, var)};
                    }

                    return {
                        std::make_shared<ngraph::op::BatchNormTraining>(epsilon, scale, bias, x)};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
