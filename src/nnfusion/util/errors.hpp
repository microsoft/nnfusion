// Microsoft (c) 2019, NNFusion Team

#pragma once

#include <stdexcept>

#include <exception>
#include <sstream>
#include <vector>
#include "logging.hpp"

namespace nnfusion
{
    namespace errors
    {
        class RuntimeError : public std::runtime_error
        {
        public:
            explicit RuntimeError(const std::string& what_arg)
                : std::runtime_error(what_arg)
            {
            }

            explicit RuntimeError(const char* what_arg)
                : std::runtime_error(what_arg)
            {
            }

            explicit RuntimeError(const std::stringstream& what_arg)
                : std::runtime_error(what_arg.str())
            {
            }
        };

        class NotSupported : public RuntimeError
        {
        public:
            NotSupported(const std::string& what_arg)
                : RuntimeError(what_arg)
            {
            }
        };

        class InvalidArgument : public std::invalid_argument
        {
        public:
            explicit InvalidArgument(const std::string& what_arg)
                : invalid_argument(what_arg)
            {
            }
        };

        class CheckError : public RuntimeError
        {
        public:
            CheckError(const std::string& what_arg)
                : RuntimeError(what_arg)
                , m_what(what_arg)
            {
            }

            CheckError(const char* what_arg)
                : RuntimeError(what_arg)
                , m_what(what_arg)
            {
            }

            const char* what() const noexcept override { return m_what.c_str(); }
        private:
            std::string m_what;
        };

        struct NullPointer : CheckError
        {
            explicit NullPointer(const std::string& what_arg)
                : CheckError(what_arg)
            {
            }
        };

        ///
        /// Helper class for failed condition check. Callers should not instantiate this class directly.
        /// This class is meant to be wrapped with a macro like CHECK, DCHECK. This class provides
        /// two main facilities: (1) an ostream accessible via get_stream(), to which a detailed
        /// error explanation can be written; and (2) throws an exception of type T when the
        /// CheckHelper is destructed.
        ///
        ///
        /// Typical usage is via a wrapper around the NNFUSION_CHECK_STREAM macro:
        ///
        ///    class MyException : public CheckError;
        ///
        ///    #define MY_CHECK(cond) NNFUSION_CHECK_STREAM(::nnfusion::errors::MyException, cond)
        ///
        ///    ...
        ///
        ///    MY_CHECK(42 != 43) << "Uh-oh. " << 42 << " is not " << 43 << ".";
        ///
        /// If the check fails, it will throw a CompileError exception with a what() string of:
        ///
        ///   Check '42 != 43' failed at foo.cpp:123:
        ///   Uh-oh. 42 is not 43.
        ///
        ///
        /// CheckHelper also provides support for tagging the exception with a "location" string,
        /// reflecting things like the op that was being processed when the error occurred. For
        /// example:
        ///
        ///   class CompileError : public CheckError;
        ///
        ///   #define COMPILE_CHECK(node,cond)                                       <backslash>
        ///      NNFUSION_CHECK_STREAM_WITH_LOC(::nnfusion::errors::CompileError, cond,          <backslash>
        ///                                    "While compiling node " + node->name())
        ///
        ///   ...
        ///
        ///   COMPILE_CHECK(node, node->get_users().size != 0) << "Node has no users";
        ///
        /// If the check fails, it will throw a CompileError exception with a what() string
        /// similar to:
        ///
        ///   While compiling node Add_123:
        ///   Check 'node->get_users().size != 0' failed at foo.cpp:123:
        ///   Node has no users
        ///
        template <class T>
        class CheckHelper
        {
        public:
            CheckHelper(const std::string& file,
                        int line,
                        const std::string& check_expression = "",
                        const std::string& location_info = "")
                : m_file(file)
                , m_line(line)
                , m_check_expression(check_expression)
                , m_location_info(location_info)
            {
            }
            ~CheckHelper() noexcept(false)
            {
                // If stack unwinding is already in progress, do not double-throw.
                if (!std::uncaught_exception())
                {
                    std::stringstream ss;
                    if (!m_location_info.empty())
                    {
                        ss << m_location_info << ":" << std::endl;
                    }

                    if (m_check_expression.empty())
                    {
                        ss << "Failure ";
                    }
                    else
                    {
                        ss << "Check failed: '" << m_check_expression << "' ";
                    }

                    ss << "at " << m_file << ":" << m_line << ":" << std::endl;

                    std::string explanation = m_stream.str();
                    if (explanation.empty())
                    {
                        explanation = "(no explanation given)";
                    }
                    ss << explanation;
                    LOG(ERROR) << ss.str();

                    throw T(ss.str());
                }
            }
            /// Returns an ostream to which additional error details can be written. The returned
            /// stream has the lifetime of the CheckHelper.
            std::ostream& get_stream() { return m_stream; }
        private:
            std::stringstream m_stream;
            std::string m_file;
            int m_line;
            std::string m_check_expression;
            std::string m_location_info;
        };

        ///
        /// Class that returns a dummy ostream to absorb error strings for non-failed check.
        /// This is cheaper to construct than CheckHelper, so the macros will produce a
        /// DummyCheckHelper in lieu of an CheckHelper if the condition is true.
        ///
        class DummyCheckHelper
        {
        public:
            /// Returns an ostream to which additional error details can be written. Anything written
            /// to this stream will be ignored. The returned stream has the lifetime of the
            /// DummyCheckHelper.
            std::ostream& get_stream() { return m_stream; }
        private:
            std::stringstream m_stream;
        };
    }
}

/// Check condition "cond" with an exception class of "T", at location "loc".
#define NNFUSION_CHECK_STREAM_WITH_LOC(T, cond, loc)                                               \
    ((cond) ? ::nnfusion::errors::DummyCheckHelper().get_stream()                                  \
            : ::nnfusion::errors::CheckHelper<T>(__FILE__, __LINE__, #cond, loc).get_stream())
/// Check condition "cond" with an exception class of "T", and no location specified.
#define NNFUSION_CHECK_STREAM(T, cond)                                                             \
    ((cond) ? ::nnfusion::errors::DummyCheckHelper().get_stream()                                  \
            : ::nnfusion::errors::CheckHelper<T>(__FILE__, __LINE__, #cond).get_stream())

/// Fails unconditionally with an exception class of "T", at location "loc".
#define NNFUSION_FAIL_STREAM_WITH_LOC(T, loc)                                                      \
    ::nnfusion::errors::CheckHelper<T>(__FILE__, __LINE__, "", loc).get_stream()
/// Fails unconditionally with an exception class of "T", and no location specified.
#define NNFUSION_FAIL_STREAM(T) ::nnfusion::errors::CheckHelper<T>(__FILE__, __LINE__).get_stream()

#define CHECK(cond) NNFUSION_CHECK_STREAM(::nnfusion::errors::CheckError, cond)
#define CHECK_FAIL() NNFUSION_FAIL_STREAM(::nnfusion::errors::CheckError)

#define CHECK_WITH_EXCEPTION(cond, T) NNFUSION_CHECK_STREAM(T, cond)
#define CHECK_FAIL_WITH_EXCEPTION(T) NNFUSION_FAIL_STREAM(T)

#define CHECK_NOT_NULLPTR(ptr_)                                                                    \
    NNFUSION_CHECK_STREAM(nnfusion::errors::NullPointer, ((ptr_) != nullptr))

#ifdef NNFUSION_DEBUG

#define DCHECK(cond) CHECK(cond)
#define DCHECK_FAIL() CHECK_FAIL()
#define DCHECK_WITH_EXCEPTION(cond, T) CHECK_WITH_EXCEPTION(cond, T)
#define DCHECK_FAIL_WITH_EXCEPTION(T) CHECK_FAIL_WITH_EXCEPTION(T)
#define DCHECK_NOT_NULLPTR(ptr_) CHECK_NOT_NULLPTR(ptr_)

#else

#define DCHECK(cond)
#define DCHECK_FAIL()
#define DCHECK_WITH_EXCEPTION(cond, T)
#define DCHECK_FAIL_WITH_EXCEPTION(T)
#define DCHECK_NOT_NULLPTR(ptr_)

#endif