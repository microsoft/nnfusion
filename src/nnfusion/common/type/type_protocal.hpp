// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cctype>
#include <cmath>
#include <cstring>
#include <map>
#include <string>

#include "../util.hpp"
#include "element_type.hpp"

namespace nnfusion
{
    namespace element
    {
        struct TypeProtocal
        {
            using New = void*();
            using NewArray = void*(size_t);
            using Copy = void(void*, void*, size_t);
            using SetElement = void(void*, size_t, const void*);
            using GetElement = void(void*, size_t, void*);
            using Delete = void(void*);
            using DeleteArray = void(void*);
            using FromString = void(void*, const std::string&);

            constexpr TypeProtocal(New* _new,
                                   NewArray* _newArray,
                                   Copy* _copy,
                                   SetElement* _setElement,
                                   GetElement* _getElement,
                                   Delete* _delete,
                                   DeleteArray* _deleteArray,
                                   FromString* _fromString) noexcept
                : f_new(_new)
                , f_newArray(_newArray)
                , f_copy(_copy)
                , f_setElement(_setElement)
                , f_getElement(_getElement)
                , f_delete(_delete)
                , f_deleteArray(_deleteArray)
                , f_fromString(_fromString)
            {
            }

            New* f_new;
            NewArray* f_newArray;
            Copy* f_copy;
            SetElement* f_setElement;
            GetElement* f_getElement;
            Delete* f_delete;
            DeleteArray* f_deleteArray;
            FromString* f_fromString;
        };

        template <typename T>
        void* defaultNew(void)
        {
            return reinterpret_cast<void*>(new T);
        }

        template <typename T>
        void* defaultNewArray(size_t n)
        {
            return reinterpret_cast<void*>(new T[n]);
        }

        template <typename T>
        void defaultCopy(void* dst, void* src, size_t n)
        {
            T* lhs = reinterpret_cast<T*>(dst);
            const T* rhs = reinterpret_cast<const T*>(src);
            std::memcpy(lhs, rhs, sizeof(T) * n);
        }

        template <typename T, typename U = T>
        void defaultSetElement(void* arr, size_t idx, const void* ele)
        {
            T* lhs = reinterpret_cast<T*>(arr);
            const U* rhs = reinterpret_cast<const U*>(ele);
            lhs[idx] = *rhs;
        }

        template <>
        void defaultSetElement<uint16_t, half>(void* arr, size_t idx, const void* ele);

        template <typename T, typename U = T>
        void defaultGetElement(void* arr, size_t idx, void* ret)
        {
            const T* rhs = reinterpret_cast<const T*>(arr);
            U* lhs = reinterpret_cast<U*>(ret);
            *lhs = rhs[idx];
        }

        template <>
        void defaultGetElement<uint16_t, half>(void* arr, size_t idx, void* ret);

        template <typename T>
        void defaultDelete(void* x)
        {
            delete reinterpret_cast<T*>(x);
        }

        template <typename T>
        void defaultDeleteArray(void* arr)
        {
            delete[] reinterpret_cast<T*>(arr);
        }

#define ERR_STATE()                                                                                \
    NNFUSION_LOG(ERROR) << "Unexpected charactor $" << c << "$ found during parsing constant"
        template <typename T>
        void fromString(void* ele, const std::string& str)
        {
            int sign_flag = 1;
            uint64_t inty = 0;
            double doub = 0., base = 1.;
            int index_sign = 1;
            uint64_t index = 0;
            int state = 1;
            for (char c : str)
            {
                if (state == 1)
                {
                    if (c == '+')
                    {
                        state = 2;
                    }
                    else if (c == '-')
                    {
                        state = 2;
                        sign_flag = -1;
                    }
                    else if (c == '.')
                    {
                        state = 3;
                    }
                    else if (isdigit(c))
                    {
                        state = 4;
                        inty = inty * 10 + c - '0';
                    }
                    else
                    {
                        ERR_STATE();
                    }
                }
                else if (state == 2)
                {
                    if (c == '.')
                    {
                        state = 3;
                    }
                    else if (isdigit(c))
                    {
                        state = 4;
                        inty = inty * 10 + c - '0';
                    }
                    else
                    {
                        ERR_STATE();
                    }
                }
                else if (state == 3)
                {
                    if (isdigit(c))
                    {
                        state = 5;
                        base /= 10.;
                        doub += base * (c - '0');
                    }
                    else
                    {
                        ERR_STATE();
                    }
                }
                else if (state == 4)
                {
                    if (c == '.')
                    {
                        state = 5;
                    }
                    else if (isdigit(c))
                    {
                        inty = inty * 10 + c - '0';
                    }
                    else
                    {
                        ERR_STATE();
                    }
                }
                else if (state == 5)
                {
                    if (c == 'e')
                    {
                        state = 6;
                    }
                    else if (isdigit(c))
                    {
                        base /= 10.;
                        inty = inty * 10 + c - '0';
                    }
                    else
                    {
                        ERR_STATE();
                    }
                }
                else if (state == 6)
                {
                    if (c == '+')
                    {
                        state = 7;
                    }
                    else if (c == '-')
                    {
                        state = 7;
                        index_sign = -1;
                    }
                    else if (isdigit(c))
                    {
                        state = 8;
                        index = index * 10 + c - '0';
                    }
                    else
                    {
                        ERR_STATE();
                    }
                }
                else if (state == 7)
                {
                    if (isdigit(c))
                    {
                        state = 8;
                        index = index * 10 + c - '0';
                    }
                    else
                    {
                        ERR_STATE();
                    }
                }
                else if (state == 8)
                {
                    if (isdigit(c))
                    {
                        index = index * 10 + c - '0';
                    }
                    else
                    {
                        ERR_STATE();
                    }
                }
                else
                {
                    ERR_STATE();
                }
            }

            T* dat = reinterpret_cast<T*>(ele);
            if (state == 4)
            {
                dat = static_cast<T>(inty);
            }
            else if (state == 5)
            {
                doub += inty;
                dat = static_cast<T>(doub);
            }
            else if (state == 8)
            {
                doub += inty;
                doub *= pow(10., index);
                dat = static_cast<T>(doub);
            }
            else
            {
                NNFUSION_LOG(ERROR) << "String ends in a not accept state.";
            }
        }
#undef ERR_STATE

        template <typename T>
        void defaultFromString(void* ele, const std::string& str)
        {
            T* dat = reinterpret_cast<T*>(ele);
            *dat = parse_string<T>(str);
        }

        template <typename T, typename U = T>
        constexpr TypeProtocal getDefaultProtocal()
        {
            return TypeProtocal(defaultNew<T>,
                                defaultNewArray<T>,
                                defaultCopy<T>,
                                defaultSetElement<T, U>,
                                defaultGetElement<T, U>,
                                defaultDelete<T>,
                                defaultDeleteArray<T>,
                                defaultFromString<U>);
        }

        constexpr TypeProtocal getEmptyProtocal()
        {
            return TypeProtocal(
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        }

        extern const std::map<Type, TypeProtocal> typeMemProto;
    } //namespace element
} //namespace nnfusion