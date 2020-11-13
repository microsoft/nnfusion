#pragma once

#include <string>
#include <vector>

#include "element_type.hpp"
#include "type_protocal.hpp"

namespace nnfusion
{
    class DataBuffer
    {
    public:
        DataBuffer() = delete;
        DataBuffer(const element::Type& _type)
            : m_type(_type)
            , m_len(0U)
            , m_data(nullptr)
        {
        }
        DataBuffer(const element::Type& _type, size_t n)
            : m_type(_type)
            , m_len(n)
            , m_data(nullptr)
        {
            resize(m_len);
        }
        ~DataBuffer();
        DataBuffer(const DataBuffer&) = delete;
        DataBuffer(DataBuffer&&);
        DataBuffer& operator=(const DataBuffer&) = delete;
        DataBuffer& operator=(DataBuffer&&);

        void resize(size_t len);
        size_t size() const;

        element::Type get_type() const;

        void setElement(size_t idx, const void* ele);
        void getElement(size_t idx, void* ret) const;

        void load(const void* src, size_t len);
        void dump(void* dst) const;
        void move_to(void** dst);

        template <typename T>
        void loadVector(const std::vector<T>& vec)
        {
            if (element::from<T>() == m_type)
            {
                resize(vec.size());
                T tmp;
                for (size_t i = 0; i < vec.size(); ++i)
                {
                    tmp = vec[i];
                    setElement(i, &tmp);
                }
            }
            else
            {
                std::vector<std::string> svec(vec.size());
                for (size_t i = 0; i < vec.size(); ++i)
                {
                    svec[i] = std::to_string(vec[i]);
                }
                loadFromStrings(svec);
            }
        }

        void loadFromStrings(const std::vector<std::string>& vec, size_t size = 0);

    private:
        void bufDelete();

    private:
        element::Type m_type;
        size_t m_len;
        void* m_data;
    };
}
