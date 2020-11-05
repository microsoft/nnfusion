#include <cstring>

#include "data_buffer.hpp"
#include "type_protocal.hpp"

using namespace nnfusion;
using element::typeMemProto;

DataBuffer::~DataBuffer() {
    bufDelete();
}

void DataBuffer::resize(size_t len) {
    bufDelete();
    if (len != 0) {
        m_data = typeMemProto.at(m_type).f_newArray(len);
        m_len = len;
    }
}

size_t DataBuffer::size() const{
    return m_len;
}
        
element::Type DataBuffer::get_type() const {
    return m_type;
}

void DataBuffer::bufDelete() {
    if (m_data != nullptr) {
        typeMemProto.at(m_type).f_deleteArray(m_data);
        m_data = nullptr;
        m_len = 0;
    }
}

DataBuffer::DataBuffer(DataBuffer &&other) {
    *this = std::move(other);
}

DataBuffer &DataBuffer::operator = (DataBuffer &&other) {
    bufDelete();
    m_type = other.m_type;
    m_len = other.m_len;
    m_data = other.m_data;
    other.m_data = nullptr;
    other.m_len = 0U;
    other.bufDelete();
}

void DataBuffer::setElement(size_t idx, const void *ele) {
    typeMemProto.at(m_type).f_setElement(m_data, idx, ele);
}

void DataBuffer::getElement(size_t idx, void *ret) const {
    typeMemProto.at(m_type).f_getElement(m_data, idx, ret);
}

void DataBuffer::load(const void *src, size_t len) {
    resize(len);
    memcpy(m_data, src, len * m_type.size());
}

void DataBuffer::dump(void *dst) const {
    typeMemProto.at(m_type).f_copy(dst, m_data, m_len);
}

void DataBuffer::move_to(void **dst) {
    *dst = m_data;
    m_data = nullptr;
    m_len = 0;
}
