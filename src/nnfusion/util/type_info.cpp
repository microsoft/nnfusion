// Microsoft (c) 2019, NNFusion Team

#include "type_info.hpp"

const nnfusion::TypeInfo::TypeDispatch nnfusion::TypeInfo::dispatcher{
    {"char", std::make_shared<nnfusion::TypeInfo_Impl<char>>()},
    {"float", std::make_shared<nnfusion::TypeInfo_Impl<float>>()},
    {"double", std::make_shared<nnfusion::TypeInfo_Impl<double>>()},
    {"int8_t", std::make_shared<nnfusion::TypeInfo_Impl<int8_t>>()},
    {"int16_t", std::make_shared<nnfusion::TypeInfo_Impl<int16_t>>()},
    {"int32_t", std::make_shared<nnfusion::TypeInfo_Impl<int32_t>>()},
    {"int64_t", std::make_shared<nnfusion::TypeInfo_Impl<int64_t>>()},
    {"uint8_t", std::make_shared<nnfusion::TypeInfo_Impl<uint8_t>>()},
    {"uint16_t", std::make_shared<nnfusion::TypeInfo_Impl<uint16_t>>()},
    {"uint32_t", std::make_shared<nnfusion::TypeInfo_Impl<uint32_t>>()},
    {"uint64_t", std::make_shared<nnfusion::TypeInfo_Impl<uint64_t>>()}};
