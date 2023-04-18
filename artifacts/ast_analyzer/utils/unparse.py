import astunparse

def unparse_ast_list(stmts):
    st = ""
    for stmt in stmts:
        stmt_st = astunparse.unparse(stmt)
        if len(stmt_st) > 1 and stmt_st[-1] == '\n':
            stmt_st = stmt_st[:-1]
        st = st + stmt_st
    return st
