#include <optional>
#include <variant>

#include "../../../arith/ir_mutator_with_analyzer.h"
#include "../utils.h"

namespace tvm {
namespace tir {

void TransformLoop(ScheduleState self, const StmtSRef& block_sref, int ndim,
                          const IndexMap& index_map) {
  const BlockNode* block_ptr = TVM_SREF_TO_BLOCK(block_sref);
  const Block& block = GetRef<Block>(block_ptr);
  arith::Analyzer analyzer;

  // Step 1: Collect outer loops and loop vars
  Array<StmtSRef> loops = GetLoops(block_sref);  // outer loops of the block
  ICHECK(ndim >= 1 && ndim <= static_cast<int>(loops.size()));
  int loop_start_idx = static_cast<int>(loops.size()) - ndim;
  StmtSRef scope_sref = loops[loop_start_idx];

  Array<PrimExpr> loop_vars;
  Array<Range> loop_ranges;
  for (size_t i = loop_start_idx; i < loops.size(); i++) {
    CheckLoopStartsWithZero(self, loops[i], &analyzer);
    const tvm::tir::ForNode* for_node = loops[i]->StmtAs<ForNode>();
    loop_vars.push_back(for_node->loop_var);
    loop_ranges.push_back({for_node->min, for_node->extent});
  }
  auto inverse_index_map = index_map.Inverse(loop_ranges);
  auto new_loop_ranges = index_map->MapRanges(loop_ranges, &analyzer);
  auto new_loop_vars = inverse_index_map->MapIndices(loop_vars, &analyzer);
  CHECK_EQ(new_loop_vars.size(), ndim);

  tvm::runtime::Map<tvm::tir::Var, tvm::PrimExpr> inverse_map;

  for (size_t i = 0; i < loop_vars.size(); i++) {
    inverse_map.Set(Downcast<Var>(loop_vars[i]), new_loop_vars[i]);
  }
  BlockRealize block_realize = GetBlockRealize(self, block_sref);

  Stmt body = Substitute(block_realize, inverse_map);

  for (int i = ndim - 1; i >= 0; --i) {
    body = For(Downcast<Var>(loop_vars[i]), 0, new_loop_ranges[i]->extent, ForKind::kSerial,
               std::move(body));
  }

  self->Replace(scope_sref, body, {});
}

}  // namespace tir
}  // namespace tvm

