from   chainer.utils import type_check
import math

from   ast_analyzer.shape_inference import utils

__all__ = [ 'ShapeElem'
          , 'wrap_shape'
          , 'unwrap_shape'
          , 'is_incomplete_shape'
          , 'copy_ShapeElem'
          , 'size_of_ShapeElem'
          , 'unify_shape'
          , 'join_shape'
          , 'is_subshape'
          ]


unaryops = {
        '-'    : (6, lambda x: -x),
        'ceil' : (6, math.ceil),
        'abs'  : (6, abs),
        'floor' : (6, math.floor),
        }

binops = {
        '+'  : (4, lambda x, y: x + y),
        '-'  : (4, lambda x, y: x - y),
        '*'  : (5, lambda x, y: x * y),
        '/'  : (5, lambda x, y: x / y),
        '//' : (5, lambda x, y: x // y),
        '%'  : (5, lambda x, y: x % y),
        }

def _flip(func):
    return (lambda x, y: func(y, x))

def _make_unaryop(term, symbol):
    priority, func = unaryops[symbol]

    if term.value is None:
        return term
    expr = type_check.UnaryOperator(priority, term.expr, symbol, func)
    return ShapeElem(func(term.value), expr=expr)

def _make_binop(lhs, rhs, symbol):
    priority, func = binops[symbol]

    if not isinstance(rhs, ShapeElem):
        if lhs.value is None:
            return ShapeElem(None)
        expr = type_check.BinaryOperator(
                priority, lhs.expr, type_check.Constant(rhs), symbol, func)
        return ShapeElem(func(lhs.value, rhs), expr=expr)

    if not isinstance(lhs, ShapeElem):
        if rhs.value is None:
            return ShapeElem(None)
        expr = type_check.BinaryOperator(
                priority, type_check.Constant(lhs), rhs.expr, symbol, func)
        return ShapeElem(func(lhs, rhs.value), expr=expr)

    if lhs.value is None or rhs.value is None:
        return ShapeElem(None)
    expr = type_check.BinaryOperator(
            priority, lhs.expr, rhs.expr, symbol, func)
    return ShapeElem(func(lhs.value, rhs.value), expr=expr)


def _make_binop_expr(lhs_expr, rhs_expr, symbol):
    priority, func = binops[symbol]
    return type_check.BinaryOperator(
            priority, lhs_expr, rhs_expr, symbol, func)

def _try_eval(expr):
    try:
        return expr.eval()
    except Exception:
        return None


def simplify(expr):
    n = _try_eval(expr)
    if n is not None:
        return type_check.Constant(n)

    if isinstance(expr, type_check.BinaryOperator):
        rhs_value = _try_eval(expr.rhs)
        if rhs_value is not None:
            if (expr.exp == '+' or expr.exp == '-') and rhs_value == 0:
                return simplify(expr.lhs)

            if expr.exp == '+' and rhs_value < 0:
                expr_rhs = type_check.Constant(- rhs_value)
                return simplify(_make_binop_expr(expr.lhs, expr_rhs, '-'))

            if expr.exp == '-' and rhs_value < 0:
                expr_rhs = type_check.Constant(- rhs_value)
                return simplify(_make_binop_expr(expr.lhs, expr_rhs, '+'))

            if (expr.exp == '*' or expr.exp == '/' or expr.exp == '//') and rhs_value == 1:
                return simplify(expr.lhs)

        lhs_value = _try_eval(expr.lhs)
        if lhs_value is not None:
            if expr.exp == '+' and lhs_value == 0:
                return simplify(expr.rhs)

        if isinstance(expr.lhs, type_check.BinaryOperator) and \
                expr.lhs.priority == expr.priority:
            if expr.lhs.exp == '+' or expr.lhs.exp == '*':
                expr_exp = expr.exp
            elif expr.lhs.exp == '-':
                if expr.exp == '+':
                    expr_exp = '-'
                else:
                    expr_exp = '+'
            else:
                assert False

            _, expr_func = binops[expr_exp]
            expr_rhs = type_check.BinaryOperator(expr.priority,
                    expr.lhs.rhs, expr.rhs, expr_exp, expr_func)
            expr_rhs = simplify(expr_rhs)
            if isinstance(expr_rhs, type_check.Constant):
                return simplify(type_check.BinaryOperator(expr.lhs.priority,
                    expr.lhs.lhs, expr_rhs, expr.lhs.exp, expr.lhs.func))

        expr.lhs = simplify(expr.lhs)
        expr.rhs = simplify(expr.rhs)

    # if isinstance(expr, type_check.UnaryOperator):
    #     expr.term = simplify(expr.term)
    return expr


class ShapeElem():
    def __init__(self, value_or_name, expr=None):
        assert type(value_or_name) in [int, float, str, type(None)]
        if isinstance(value_or_name, str):
            # name (given via type hints)
            self.value = value_or_name
            self.expr = None
        else:
            # value
            self.value = value_or_name
            if expr is None:
                self.expr = type_check.Constant(value_or_name)
            else:
                self.expr = simplify(expr)

    def __str__(self):
        if isinstance(self.expr, type_check.Constant):
            return str(self.value)
        if self.is_null() and self.expr is None:
            return "None"
        if self.expr.__str__() is None:
            return str(self.value)
        else:
            return "{} ({})".format(str(self.value), str(self.expr))

    def __repr__(self):
        return self.__str__()

    def __neg__(self):
        return _make_unaryop(self, '-')
    def __ceil__(self):
        return _make_unaryop(self, 'ceil')
    def __abs__(self):
        return _make_unaryop(self, 'abs')
    def __floor__(self):
        return _make_unaryop(self, 'floor')

    def __add__(self, other):
        return _make_binop(self, other, '+')
    def __sub__(self, other):
        return _make_binop(self, other, '-')
    def __mul__(self, other):
        return _make_binop(self, other, '*')
    def __truediv__(self, other):
        return _make_binop(self, other, '/')
    def __floordiv__(self, other):
        return _make_binop(self, other, '//')
    def __mod__(self, other):
        return _make_binop(self, other, '%')

    def __gt__(self, other):
        if self.value is None or other.value is None:
            return True
        return self.value > other.value

    def __lt__(self, other):
        if self.value is None or other.value is None:
            return True
        return self.value < other.value


    __iadd__ = __add__
    __isub__ = __sub__
    __imul__ = __mul__
    __itruediv__ = __truediv__
    __ifloordiv__ = __floordiv__
    __imod__ = __mod__

    __radd__ = _flip(__add__)
    __rsub__ = _flip(__sub__)
    __rmul__ = _flip(__mul__)
    __rtruediv__ = _flip(__truediv__)
    __rfloordiv__ = _flip(__floordiv__)
    __rmod__ = __mod__

    def __eq__(self, other):
        # XXX: equality against None should always be true
        if self.value is None:
            return True

        if isinstance(other, ShapeElem):
            if other.value is None:
                return True
            return self.value == other.value
        else:
            return self.value == other

    def is_null(self):
        return self.value is None

    def has_value(self):
        return self.value is not None

    def get_value(self):
        return self.value

    def __deepcopy__(self, memo):
        # WARNING: not a real deepcopy!!
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        return result


def wrap_shape(shape_seq): # Tuple[int or ShapeElem] -> Tuple[ShapeElem]
    return tuple([i if isinstance(i, ShapeElem) else ShapeElem(i) for i in shape_seq])

def unwrap_shape(shape_seq, keep_None = False):
    if keep_None and is_incomplete_shape(shape_seq):
        return tuple([s.value if s.has_value() else s.expr.name for s in shape_seq])
    else:
        return tuple([s.value if s.has_value() else 1 for s in shape_seq])

def is_incomplete_shape(shape_seq):
    return any([not s.has_value() for s in shape_seq])

def copy_ShapeElem(e):
    return ShapeElem(e.value, expr=e.expr)


def size_of_ShapeElem(e):
    if isinstance(e.expr, type_check.BinaryOperator):
        return size_of_ShapeElem(e.lhs) + size_of_ShapeElem(e.rhs)
    if isinstance(e.expr, type_check.UnaryOperator):
        return size_of_ShapeElem(e.term)
    if isinstance(e.expr, type_check.Variable):
        return 1
    return 0


def unify_shape(shape1, shape2):
    for e1, e2 in zip(shape1, shape2):
        if e1.value != e2.value:
            e1.value = e2.value = None
    return shape1


def join_shape(shape1, shape2):
    ret = [None for _ in shape1]
    for i, (e1, e2) in enumerate(zip(shape1, shape2)):
        if e1.value == e2.value:
            ret[i] = e1.value
    return ret


def is_subshape(shape1, shape2):
    def is_subShapeElem(e1, e2):
        return e2.value is None or e1.value == e2.value
    return all([is_subShapeElem(e1, e2) for e1, e2 in zip(shape1, shape2)])



