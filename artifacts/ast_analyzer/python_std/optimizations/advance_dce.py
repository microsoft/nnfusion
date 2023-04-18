from ast_analyzer.grad import annotate, transformers
from ast_analyzer.grad import annotations as anno
import gast
import astunparse


def advance_dce(node):
    """Perform a simple form of dead code elimination on a Python AST.

    This method performs reaching definitions analysis on all function
    definitions. It then looks for the definition of variables that are not used
    elsewhere and removes those definitions.

    This function takes into consideration push and pop statements; if a pop
    statement is removed, it will also try to remove the accompanying push
    statement. Note that this *requires dead code elimination to be performed on
    the primal and adjoint simultaneously*.

    Args:
      node: The AST to optimize.

    Returns:
      The optimized AST.
    """

    func_ty = anno.getanno(node.body[0], 'type')
    # num_input = len(func_ty.argty)
    num_result = 1
    to_remove = annotate.unused(node.body[1], True)
    first_var = 1 if node.body[1].args.args[0].id == 'self' else 0
    for arg in node.body[1].args.args[:first_var + num_result]:
        if arg in to_remove:
            to_remove.remove(arg)
    transformers.Remove(to_remove).visit(node.body[1])
    # raise NotImplementedError
    anno.clearanno(node.body[1])

    removed_args = set(x.id for x in to_remove if isinstance(x, gast.Name))

    node.body[0].body[-1].value.elts = list(
        x for x in node.body[0].body[-1].value.elts if x.id not in removed_args or anno.hasanno(x, 'origin_ret')
    )

    to_remove = annotate.unused(node.body[0])
    transformers.Remove(to_remove).visit(node.body[0])
    anno.clearanno(node.body[0])

    # print("[Advance DCE]", astunparse.unparse(node))
