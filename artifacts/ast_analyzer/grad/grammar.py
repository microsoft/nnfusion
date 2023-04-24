"""Classifications of AST nodes."""
from __future__ import absolute_import
import gast

# LITERALS are representated by gast.Constant

CONTROL_FLOW = (gast.For, gast.AsyncFor, gast.While, gast.If, gast.Try,
                gast.Break, gast.Continue)

COMPOUND_STATEMENTS = (
    gast.FunctionDef,
    gast.ClassDef,
    gast.For,
    gast.While,
    gast.If,
    gast.With,
    gast.Try,
    gast.AsyncFunctionDef,
    gast.AsyncFor,
    gast.AsyncWith
)

SIMPLE_STATEMENTS = (
    gast.Return,
    gast.Delete,
    gast.Assign,
    gast.AugAssign,
    gast.Raise,
    gast.Assert,
    gast.Import,
    gast.ImportFrom,
    gast.Global,
    gast.Nonlocal,
    gast.Expr,
    gast.Pass,
    gast.Break,
    gast.Continue
)

STATEMENTS = COMPOUND_STATEMENTS + SIMPLE_STATEMENTS

BLOCKS = (
    (gast.Module, 'body'),
    (gast.FunctionDef, 'body'),
    (gast.AsyncFunctionDef, 'body'),
    (gast.For, 'body'),
    (gast.For, 'orelse'),
    (gast.AsyncFor, 'body'),
    (gast.AsyncFor, 'orelse'),
    (gast.While, 'body'),
    (gast.While, 'orelse'),
    (gast.If, 'body'),
    (gast.If, 'orelse'),
)
