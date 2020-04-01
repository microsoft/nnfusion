// Microsoft (c) 2019, NNFusion Team

#pragma once

#include <memory>
#include <vector>

namespace nnfusion
{
    namespace graph
    {
        class GNode;

        /// \brief Zero or more nodes.
        class GNodeVector : public std::vector<std::shared_ptr<GNode>>
        {
        public:
            GNodeVector(const std::initializer_list<std::shared_ptr<GNode>>& gnodes)
                : std::vector<std::shared_ptr<GNode>>(gnodes)
            {
            }

            GNodeVector(const std::vector<std::shared_ptr<GNode>>& gnodes)
                : std::vector<std::shared_ptr<GNode>>(gnodes)
            {
            }

            GNodeVector(const GNodeVector& gnodes)
                : std::vector<std::shared_ptr<GNode>>(gnodes)
            {
            }

            GNodeVector(size_t size)
                : std::vector<std::shared_ptr<GNode>>(size)
            {
            }

            GNodeVector& operator=(const GNodeVector& other) = default;

            GNodeVector() {}
        };
    }
}
