import gast

# [NOT tested]
class TwoNodeVisitor(object):
    """
    A node visitor base class that walks the abstract syntax tree and calls a
    visitor function for every node found.  This function may return a value
    which is forwarded by the `visit` method.

    This class is meant to be subclassed, with the subclass adding visitor
    methods.

    Per default the visitor functions for the nodes are ``'visit_'`` +
    class name of the node.  So a `TryFinally` node visit function would
    be `visit_TryFinally`.  This behavior can be changed by overriding
    the `visit` method.  If no visitor function exists for a node
    (return value `None`) the `generic_visit` visitor is used instead.

    Don't use the `NodeVisitor` if you want to apply changes to nodes during
    traversing.  For this a special visitor exists (`NodeTransformer`) that
    allows modifications.
    """

    def visit(self, node1, node2):
        """Visit a node."""
        assert(type(node1) == type(node2))
        method = 'visit_' + node1.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node1, node2)

    def generic_visit(self, node1, node2):
        """Called if no explicit visitor function exists for a node."""
        attrs = set(node1._fields).intersection(set(node2._fields))
        for field in node1._fields:
            value = getattr(node1, field, None)
            if isinstance(value, list) or isinstance(value, gast.AST):
                assert field in attrs, "node1 field not match: " + field
        for field in node2._fields:
            value = getattr(node2, field, None)
            if isinstance(value, list) or isinstance(value, gast.AST):
                assert field in attrs, "node2 field not match: " + field

        for field in node1._fields:
            value1 = getattr(node1, field, None)
            value2 = getattr(node2, field, None)
            if isinstance(value1, list) and isinstance(value2, list):
                assert(len(value1) == len(value2))
                for item1, item2 in zip(value1, value2):
                    assert(isinstance(item1, gast.AST) == isinstance(item2, gast.AST))
                    if isinstance(item1, gast.AST):
                        self.visit(item1, item2)
            elif isinstance(value1, list) and isinstance(value2, list):
                assert False, "list field not match: " + field
            elif isinstance(value1, gast.AST) and isinstance(value2, gast.AST):
                self.visit(value1, value2)
            elif isinstance(value1, gast.AST) or isinstance(value2, gast.AST):
                assert False, "ast field not match: " + field
