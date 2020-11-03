#pragma once
#include <element_type.hpp>
#include <map>

namespace nnfusion {
    namespace element {
        struct TypeProtocal {
            using New = void*();
            using NewArray = void*(size_t);
            using Copy = void(void *, void *, size_t);
            using SetElement = void(void *, size_t, void *);
            using GetElement = void(void *, size_t, void *);
            using Delete = void(void*);
            using DeleteArray = void(void*); 

            constexpr TypeProtocal(
                New *_new,
                NewArray *_newArray,
                Copy *_copy,
                SetElement *_setElement,
                GetElement *_getElement,
                Delete *_delete,
                DeleteArray *_deleteArray
            ) noexcept
            :   f_new(_new),
                f_newArray(_newArray),
                f_copy(_copy),
                f_setElement(_setElement),
                f_getElement(_getElement),
                f_delete(_delete),
                f_deleteArray(_delete) {}

            New *f_new;
            NewArray *f_newArray;
            Copy *f_copy;
            SetElement *f_setElement;
            GetElement *f_getElement;
            Delete *f_delete;
            DeleteArray *f_deleteArray;
        };

        template <typename T>
        void *defaultNew(void) {
            return reinterpret_cast<void *>(new T);
        }

        template <typename T>
        void *defaultNewArray(size_t n) {
            return reinterpret_cast<void *>(new T[n]);
        }

        template <typename T>
        void defaultCopy(void *dst, void *src, size_t n) {
            T *lhs = reinterpret_cast<T*>(dst);
            const T *rhs = reinterpret_cast<const T*>(src);
            for (int i = 0; i < n; ++i) {
                lhs[i] = rhs[i];
            }
        }

        template <typename T, typename U=T>
        void defaultSetElement(void *arr, size_t idx, void *ele) {
            T *lhs = reinterpret_cast<T*>(arr);
            const U *rhs = reinterpret_cast<const U*>(ele);
            lhs[idx] = rhs;
        }

        template <>
        void defaultSetElement<uint16_t, half>(void *arr, size_t idx, void *ele);

        template <typename T, typename U=T>
        void defaultGetElement(void *arr, size_t idx, void *ret) {
            const U *rhs = reinterpret_cast<const U*>(arr);
            T *lhs = reinterpret_cast<T*>(ret);
            &lhs = rhs[idx];
        }

        template <>
        void defaultGetElement<uint16_t, half>(void *arr, size_t idx, void *ret);

        template <typename T>
        void defaultDelete(void *x) {
            delete reinterpret_cast<T*>(x);
        }

        template <typename T>
        void defaultDeleteArray(void *arr) {
            delete[] reinterpret_cast<T*>(arr);
        }

        template <typename T, typename U=T>
        constexpr TypeProtocal getDefaultProtocal() {
            return TypeProtocal(
                defaultNew<T>,
                defaultNewArray<T>,
                defaultCopy<T, U>,
                defaultSetElement<T, U>,
                defaultGetElement<T, U>,
                defaultDelete<T>,
                defaultDeleteArray<T>
            );
        }

        constexpr TypeProtocal getEmptyProtocal() {
            return TypeProtocal(
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr
            );
        }

        extern std::map<Type, TypeProtocal> typeMemProto;
    } //namespace element
} //namespace nnfusion