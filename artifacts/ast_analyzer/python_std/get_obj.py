import gast as ast
def get_obj(obj, node):
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == 'self':
            return getattr(obj, node.attr)
        else:
            return None
    else:
        return None