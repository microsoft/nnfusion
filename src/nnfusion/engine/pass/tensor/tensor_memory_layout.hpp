// Microsoft (c) 2019, NNFUSION TEAM
#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/engine/interpreter.hpp"
#include "nnfusion/engine/op.hpp"

namespace nnfusion
{
    namespace pass
    {
        ///\brief Obsoleted MemoryLayout pass doesn't support tensor allcoated
        /// by KernelEmitter, thus this class is to fix the problem.
        /// Follows this order:
        /// KernelSelected -> AssignTensorMemoryLayout -> Codegen
        class AssignTensorMemoryLayout : public IInterpreterPass
        {
        public:
            AssignTensorMemoryLayout(size_t alignment = 64, bool disable_memory_sharing = false)
                : m_alignment(alignment)
                , m_disable_memory_sharing(disable_memory_sharing)
            {
            }

            bool run(std::shared_ptr<InterpreterContext> ctx,
                     std::shared_ptr<TranslationUnit> tu) override;

        private:
            size_t m_alignment;
            bool m_disable_memory_sharing;
        };
    }
}