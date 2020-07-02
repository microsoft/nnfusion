#include "attribute.hpp"

namespace nnfusion
{
    namespace ir
    {
        TagProxy Tags::operator[](Symbol sym) { return TagProxy(this, sym); }
        template <>
        void TagProxy::operator=<char*>(char* str)
        {
            std::string c_str(str);
            this->set(c_str);
        }

        template <>
        void TagProxy::operator=<char[]>(char str[])
        {
            std::string c_str(str);
            this->set(c_str);
        }

        template <>
        const Tagable* TagProxy::set<char*>(char* val)
        {
            std::string c_str(val);
            return this->set(c_str);
        }

        template <>
        const Tagable* TagProxy::set<char[]>(char val[])
        {
            std::string c_str(val);
            return this->set(c_str);
        }
    }
}