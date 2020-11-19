// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <map>

#include "element_type.hpp"
#include "type_protocal.hpp"

using namespace nnfusion;

const std::map<element::Type, element::TypeProtocal> element::typeMemProto = {
    {element::dynamic, element::getEmptyProtocal()},
    {element::boolean, element::getDefaultProtocal<bool>()},
    {element::character, element::getDefaultProtocal<char>()},
    {element::bf16, element::getDefaultProtocal<char>()}, // TODO: verify what to do with bf16
    {element::f16,
     element::getDefaultProtocal<uint16_t, element::half>()}, // TODO: verify what to do with fp16
    {element::f32, element::getDefaultProtocal<float>()},
    {element::f64, element::getDefaultProtocal<double>()},
    {element::i8, element::getDefaultProtocal<int8_t>()},
    {element::i16, element::getDefaultProtocal<int16_t>()},
    {element::i32, element::getDefaultProtocal<int32_t>()},
    {element::i64, element::getDefaultProtocal<int64_t>()},
    {element::u8, element::getDefaultProtocal<uint8_t>()},
    {element::u16, element::getDefaultProtocal<uint16_t>()},
    {element::u32, element::getDefaultProtocal<uint32_t>()},
    {element::u64, element::getDefaultProtocal<uint64_t>()},
};

/*
template <> 
void element::defaultCopy<uint16_t, element::half>(void *dst, void *src, size_t n) {
    uint16_t *lhs = reinterpret_cast<uint16_t*>(dst);
    const half *rhs = reinterpret_cast<const half*>(src);
    for (int i = 0; i < n; ++i) {
        lhs[i] = rhs[i].get_raw_data();
    }
}
*/

template <>
void element::defaultSetElement<uint16_t, element::half>(void* arr, size_t idx, const void* ele)
{
    uint16_t* lhs = reinterpret_cast<uint16_t*>(arr);
    const half* rhs = reinterpret_cast<const half*>(ele);
    lhs[idx] = rhs->get_raw_data();
}

template <>
void element::defaultGetElement<uint16_t, element::half>(void* arr, size_t idx, void* ret)
{
    const uint16_t* rhs = reinterpret_cast<const uint16_t*>(arr);
    half* lhs = reinterpret_cast<half*>(ret);
    lhs->set_raw_data(rhs[idx]);
}
