// Microsoft (c) 2019, NNFusion Team

#include "generic_op.hpp"

namespace nnfusion
{
    namespace op
    {
        std::unordered_map<std::string, OpConfig>& get_op_configs()
        {
            static std::unordered_map<std::string, OpConfig> __op_configs;
            return __op_configs;
        }
    } // namespace op
} // namespace nnfusion
