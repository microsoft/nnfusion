// Microsoft (c) 2019, Wenxiang
// Metagraph IR, which is to guide the codegen procedcure.
// This IR is based on ONNIX::ir's interface, but
// Instructions has attribute, namespace, and tag

#pragma once

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_set>
#include <vector>

#include "nnfusion/util/util.hpp"

namespace nnfusion
{
    namespace ir
    {
        using Symbol = std::string;

        struct AttributeValue
        {
            AttributeValue(Symbol name)
                : name(name)
                , type_hash(0)
            {
            }
            AttributeValue(Symbol name, size_t type_hash)
                : name(name)
                , type_hash(type_hash)
            {
            }
            using Ptr = std::unique_ptr<AttributeValue>;
            Symbol name;
            size_t type_hash;
            virtual Ptr clone() const = 0;
            virtual ~AttributeValue() = default;

            template <typename T>
            bool is_a() const
            {
                if (type_hash == 0)
                    LOG(WARNING) << "Attribute value type hash code was set to zero,"
                                    " this will ignore type check.";
                return (type_hash == 0) || (typeid(T).hash_code() == type_hash);
            }
        };

        template <typename T>
        struct ScalarAttributeValue final : public AttributeValue
        {
            using ConstructorType = const T&;
            using ValueType = T;
            ScalarAttributeValue(Symbol name, ConstructorType value_)
                : AttributeValue(name)
                , value_(value_)
            {
                // RTTI binding of type hash code, is there any better solution?
                type_hash = typeid(T).hash_code();
            }
            ValueType& value() { return value_; }
            virtual Ptr clone() const override
            {
                return Ptr(new ScalarAttributeValue(name, value_));
            }

            bool check_type() { return is_a<T>(); }
        private:
            ValueType value_;
        };

        template <typename T>
        struct VectorAttributeValue final : public AttributeValue
        {
            using ConstructorType = const std::vector<T>&&;
            using ValueType = std::vector<T>;
            VectorAttributeValue(Symbol name, ConstructorType value_)
                : AttributeValue(name)
                , value_(std::move(value_))
            {
                type_hash = typeid(T).hash_code();
            }
            ValueType& value() { return value_; }
            virtual std::unique_ptr<AttributeValue> clone() const override
            {
                auto copy = value_;
                return Ptr(new VectorAttributeValue(name, std::move(copy)));
            }
            bool check_type() { return is_a<T>(); }
        private:
            ValueType value_;
        };

        using FloatAttr = ScalarAttributeValue<double>;
        using FloatsAttr = VectorAttributeValue<double>;
        using IntAttr = ScalarAttributeValue<int64_t>;
        using IntsAttr = VectorAttributeValue<int64_t>;
        using StringAttr = ScalarAttributeValue<std::string>;
        using StringsAttr = VectorAttributeValue<std::string>;
        //using TensorAttr = ScalarAttributeValue<nnfusion::TensorWrapper>;
        //using TensorsAttr = VectorAttributeValue<nnfusion::TensorWrapper>;

        class Attributes
        {
        public:
            using Pointer = std::shared_ptr<Attributes>;
            Attributes() {}
            void copyAttributes(const Attributes& rhs)
            {
                values_.clear();
                values_.reserve(rhs.values_.size());
                for (auto& i : rhs.values_)
                {
                    values_.push_back(i->clone());
                }
            }
            bool hasAttribute(Symbol name) const { return find(name, false) != values_.end(); }
            Attributes* removeAttribute(Symbol name)
            {
                values_.erase(find(name, true));
                return this;
            }
            bool hasAttributes() const { return values_.size() > 0; }
            // The names are returned in order, since name actually is the index.
            std::vector<Symbol> attributeNames() const
            {
                std::vector<Symbol> names;
                names.reserve(values_.size());
                for (auto& a : values_)
                    names.push_back(a->name);
                return names;
            }

#define CREATE_ACCESSOR(Kind, method)                                                              \
    Attributes* method##_(Symbol name, Kind##Attr::ConstructorType v)                              \
    {                                                                                              \
        return set<Kind##Attr>(name, std::forward<Kind##Attr::ConstructorType>(v));                \
    }                                                                                              \
    const Kind##Attr::ValueType& method(Symbol name) const { return get<Kind##Attr>(name); }
            CREATE_ACCESSOR(Float, f)
            CREATE_ACCESSOR(Floats, fs)
            CREATE_ACCESSOR(String, s)
            CREATE_ACCESSOR(Strings, ss)
            CREATE_ACCESSOR(Int, i)
            CREATE_ACCESSOR(Ints, is)
//CREATE_ACCESSOR(Tensor, t)
//CREATE_ACCESSOR(Tensors, ts)

#undef CREATE_ACCESSOR

        protected:
            template <typename T>
            Attributes* set(Symbol name, typename T::ConstructorType v)
            {
                auto it = find(name, false);
                auto nv = AVPtr(new T(name, std::forward<typename T::ConstructorType>(v)));
                if (it == values_.end())
                {
                    values_.push_back(std::move(nv));
                }
                else
                {
                    *it = std::move(nv);
                }
                return this;
            }

            template <typename T>
            bool is_a(Symbol name) const
            {
                auto it = find(name, true);
                T* child = static_cast<T*>(it->get());
                return child->check_type();
            }

            template <typename T>
            typename T::ValueType& get(Symbol name) const
            {
                auto it = find(name, true);
                T* child = static_cast<T*>(it->get());
                using valuetype = typename T::ValueType;
                CHECK(child->check_type()) << "Try to access the value using invalid data type: "
                                           << typeid(valuetype).name() << ".";
                return child->value();
            }
            using AVPtr = AttributeValue::Ptr;
            // NB: For determinism, we use a vector rather than a hash map.  This does
            // mean that lookups are O(n), so you shouldn't use Attributes to store
            // a big pile of messages.
            std::vector<AVPtr> values_;
            using iterator = std::vector<AVPtr>::iterator;
            iterator find(Symbol name, bool required)
            {
                auto it = std::find_if(values_.begin(), values_.end(), [&](const AVPtr& v) {
                    return v->name == name;
                });
                CHECK(!required || it != values_.end()) << "The attribute is not existed.";
                return it;
            }
            using const_iterator = std::vector<AVPtr>::const_iterator;
            const_iterator find(Symbol name, bool required) const
            {
                auto it = std::find_if(values_.begin(), values_.end(), [&](const AVPtr& v) {
                    return v->name == name;
                });
                CHECK(!required || it != values_.end()) << "required undefined attribute:" << name;
                return it;
            }
        };

        class TagProxy;

        class Tagable : public Attributes
        {
        public:
            using Pointer = std::shared_ptr<Tagable>;
            template <typename T>
            Tagable* Set(Symbol name, T&& v)
            {
                set<ScalarAttributeValue<T>>(name, v);
                return this;
            }

            template <typename T>
            T& Get(Symbol name) const
            {
                return get<ScalarAttributeValue<T>>(name);
            }

            template <typename T>
            bool CheckType(Symbol name) const
            {
                return is_a<ScalarAttributeValue<T>>(name);
            }

            void copy_tags_from(const Tagable& attr)
            {
                for (auto& it : attr.values_)
                {
                    if (find(it->name, false) == values_.end())
                    {
                        // The value's uniq_ptr;
                        values_.push_back(it->clone());
                    }
                    else
                    {
                        LOG(WARNING)
                            << "Tagable interface will not copy existed item with same name.";
                    }
                }
            }

            TagProxy operator[](Symbol sym);
        };

        using Tags = Tagable;

        class TagProxy
        {
        private:
            TagProxy() {}
        public:
            TagProxy(Tagable* tags, Symbol sym)
                : _tags(tags)
                , _sym(sym)
            {
            }

            bool is_valid() { return _tags->hasAttribute(_sym); };
            template <typename T>
            bool is_valid_as()
            {
                return is_valid() && _tags->CheckType<T>(_sym);
            };

            template <typename T>
            const Tagable* set(T val)
            {
                CHECK_NOT_NULLPTR(_tags);
                _tags->Set<T>(_sym, std::move(val));
                return _tags;
            }

            template <typename T>
            const Tagable* set_rval(T&& val)
            {
                CHECK_NOT_NULLPTR(_tags);
                _tags->Set<T>(_sym, val);
                return _tags;
            }

            template <typename T>
            void operator=(T val)
            {
                set(val);
            }

            // Special type conversion assuming data is bool type;
            operator bool() { return is_valid_as<bool>() && this->as<bool>(); }
            ///\brief This will give the reference;
            ///\todo(wenxh) Need check the datatype, which should be stored at
            // ScalarAttributeValue.
            template <typename T>
            T& as()
            {
                CHECK(is_valid()) << "Tag doesn't have item who's name is: " << _sym << ".";
                return _tags->Get<T>(_sym);
            }

            ///\brief This is not deep copy.
            template <typename T>
            T clone()
            {
                CHECK(is_valid()) << "Tag doesn't have item who's name is: " << _sym << ".";
                return _tags->Get<T>(_sym);
            }

        private:
            Tagable* _tags = nullptr;
            Symbol _sym;
        };

        template <>
        void nnfusion::ir::TagProxy::operator=<char*>(char* str);
        template <>
        void nnfusion::ir::TagProxy::operator=<char[]>(char str[]);
        template <>
        const Tagable* TagProxy::set<char*>(char* val);
        template <>
        const Tagable* TagProxy::set<char[]>(char val[]);
    }
}
