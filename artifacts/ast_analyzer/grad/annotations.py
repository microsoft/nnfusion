"""Handling annotations on AST nodes."""
from __future__ import absolute_import

import gast
import copy

ANNOTATION_FIELD = '_grad_anno'
# These annotation's won't be cleared between passes
FIXED_ANNOTATIONS = set(['pop', 'push', 'add_grad', 'init_grad', 'pri', 'adj',
                         'push_func', 'pop_func', 'adjoint_var',
                         'temp_adjoint_var', 'temp_var', 'pri_call',
                         'adj_call', 'comment', 'pre_anf',
                         'store_stmt', 'restore_stmt', 'cache_ret', 'cache_arg',
                         'cache_target_ret', 'cache_target_arg', 'related_nodes', 'save_var', 'type', 'origin_ret', 'attr_name', 'manual', 'can_push', 'may_push'])


class Annotation:
    def __init__(self):
        self.annos = {}

    def __deepcopy__(self, memo):
        dpcpy = self.__class__()
        memo[id(self)] = dpcpy
        new_anno = Annotation()
        for s in self.annos:
            new_anno.annos[s] = self.annos[s]
        return new_anno

    def __str__(self):
        return "[Annotation]" + str(self.annos)

    def __repr__(self):
        return "[Annotation]" + str(self.annos)


def setanno(node, key, value, safe=True):
    annotations = getattr(node, ANNOTATION_FIELD, Annotation())
    setattr(node, ANNOTATION_FIELD, annotations)
    if safe and hasanno(node, key):
        raise ValueError('annotation already present:', key, gast.dump(node))
    annotations.annos[key] = value

    # So that the annotations survive gast_to_ast() and ast_to_gast()
    if ANNOTATION_FIELD not in node._fields:
        node._fields += (ANNOTATION_FIELD,)


def hasanno(node, key):
    annotations = getattr(node, ANNOTATION_FIELD, Annotation())
    return key in annotations.annos


def setdefaultanno(node, key, value=None):
    if not hasanno(node, key):
        setanno(node, key, value)
    return getanno(node, key)


def clearanno(node):
    for succ in gast.walk(node):
        if hasattr(succ, ANNOTATION_FIELD):
            new = Annotation()
            for anno in FIXED_ANNOTATIONS:
                if hasanno(succ, anno):
                    new.annos[anno] = getanno(succ, anno)
            setattr(succ, ANNOTATION_FIELD, new)
    return node


def getanno(node, key, default=None):
    annotations = getattr(node, ANNOTATION_FIELD, Annotation())
    if key not in annotations.annos and default is None:
        raise KeyError('Node "%s" has no annotation "%s"' % (gast.dump(node), key))
    return annotations.annos.get(key, default)


def delanno(node, key):
    annotations = getattr(node, ANNOTATION_FIELD, Annotation())
    del annotations.annos[key]
