// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <utility>
#include "../gnode.hpp"
#include "../graph.hpp"

namespace nnfusion
{
    namespace graph
    {
        class autobroadcast_incompatible_shapes : public errors::CheckError
        {
        public:
            autobroadcast_incompatible_shapes(const nnfusion::Shape& shape1,
                                              const nnfusion::Shape& shape2);

            const nnfusion::Shape& get_shape1() const;
            const nnfusion::Shape& get_shape2() const;

        private:
            const nnfusion::Shape m_shape1;
            const nnfusion::Shape m_shape2;

            static std::string error_str(const nnfusion::Shape& shape1,
                                         const nnfusion::Shape& shape2);
        };

        /// \brief Wrap two graph nodes, if necessary, to obtain values with identical shapes,
        /// using NumPy's auto-broadcast rules.
        ///
        /// The elements in the std::pair returned by this function correspond to those supplied
        /// in the std::pair provided via \p args.
        ///
        /// If \p args.first and \p args.second produce identical shapes, then the returned std::pair
        /// will have the same value as \p args.
        ///
        /// If \p args.first and \p args.second produce different shapes, then this function creates
        /// new ngraph::op::Reshape and/or ngraph::op::Broadcast nodes, as needed, to wrap
        /// \p args.first and/or \p args.second in a manner that yields values with the same shape.
        ///
        /// There are some shape combinations which the autobroadcast algoritm cannot handle.
        /// An exception is thrown when such combinations are provided to this function.
        ///
        /// \pre
        /// - \p args.first is not null
        /// - \p args.second is not null
        ///
        /// \post
        /// - The ngraph::Node objects pointed to by \p args.first and \p args.second have not been
        ///   altered by this function, except by possibly having added consumers of their values.
        ///
        /// - If an exception was not thrown, then the return value's \p first and \p second
        ///   elements point to ngraph::Node objects whose output values have the same shape.
        ///
        /// \exception nnfusion::graph::autobroadcast_incompatible_shapes
        std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>
            numpy_broadcast(const std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>& args,
                            std::shared_ptr<nnfusion::graph::Graph> graph);

        /* todo
        /// Create a new \p NodeType node, and any additional nodes required to simulate NumPy-style autobroadcast
        /// semantics.  Intended for binary operations such as "Add".
        ///
        /// \param [in] operand1_reshapeable The first operand to supply to the \p NodeType constructor.  Subject to
        ///   being wrapped with additional nodes required for autobroadcasting.  Must not be null.
        ///
        /// \param [in] operand2_reshapeable The second operand to supply to the \p NodeType constructor.  Subject to
        ///   being wrapped with additional nodes required for autobroadcasting.  Must not be null.
        ///
        /// \return The sink node of any/all nodes created by this function.  Will never be null.
        ///
        /// \exception nnfusion::graph::autobroadcast_incompatible_shapes
        template <typename NodeType>
        std::shared_ptr<NodeType>
            make_with_numpy_broadcast(const std::shared_ptr<GNode>& operand1_reshapeable,
                                      const std::shared_ptr<GNode>& operand2_reshapeable)
        {
            std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>> shaped_op1_op2 =
                numpy_broadcast({operand1_reshapeable, operand2_reshapeable});
            return std::make_shared<NodeType>(shaped_op1_op2.first, shaped_op1_op2.second);
        }

        /// Create a new \p NodeType node, and any additional nodes required to simulate NumPy-style autobroadcast
        /// semantics.  Intended for non-binary operations such as "Select", where precisely the second and third
        /// operands are subject to autobroadcast semantics.
        ///
        /// \param [in] operand1 This operand is not subject to autobraodcast logic, and will be passed as-is as
        /// the first argument to the \p NodeType constructor.
        ///
        /// \param [in] operand2_reshapeable The second operand to supply to the \p NodeType constructor.  Subject to
        ///   being wrapped with additional nodes required for autobroadcasting.  Must not be null.
        ///
        /// \param [in] operand3_reshapeable The third operand to supply to the \p NodeType constructor.  Subject to
        ///   being wrapped with additional nodes required for autobroadcasting.  Must not be null.
        ///
        /// \return The sink node of any/all nodes created by this function.  Will never be null.
        ///
        /// \exception nnfusion::graph::autobroadcast_incompatible_shapes
        template <typename NodeType>
        std::shared_ptr<NodeType>
            make_with_numpy_broadcast(const std::shared_ptr<GNode>& operand1,
                                      const std::shared_ptr<GNode>& operand2_reshapeable,
                                      const std::shared_ptr<GNode>& operand3_reshapeable)
        {
            std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>> shaped_op2_op3 =
                numpy_broadcast({operand2_reshapeable, operand3_reshapeable});
            return std::make_shared<NodeType>(
                operand1, shaped_op2_op3.first, shaped_op2_op3.second);
        }
        */

    } // namespace builder
} // namespace ngraph
