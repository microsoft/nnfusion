# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This incompetent parser has been deprecated
Basic lexer and parser needed
pip install ply
"""

from __future__ import generators
import re
import os.path
import time
import copy
import sys
import ply.yacc as yacc
import ply.lex as lex

# Python 3 compatibility
if sys.version_info.major < 3:
    STRING_TYPES = (str, unicode)
else:
    STRING_TYPES = str
    xrange = range

# tokens for lexing
reserved = {
    # keywords
    'if': 'IF',
    'else': 'ELSE',
    'for': 'FOR',
    'void': 'VOID',
    # mark
    # '*': '*',
    # ',': ',',
    # ';': ';',
    # '(': '(',
    # ')': ')',
    # '[': '[',
    # ']': ']',
    # '{': '{',
    # '}': '}',
    # data types
    'bool': 'TYPE',
    'char': 'TYPE',
    'double': 'TYPE',
    'float': 'TYPE',
    'int': 'TYPE',
    # qualifiers
    'const': 'QUALIFIER',
    '__volatile__': 'QUALIFIER',
    '__restrict__': 'QUALIFIER',
    # CUDA
    '__shared__': 'SHARED',
    '__syncthreads': 'SYNC',
    '__global__': 'GLOBAL'
}

tokens = ['ID', 'INTEGER', 'FLOAT', 'STRING', 'CHAR', 'WS', 'COMMENT1', 'COMMENT2', 'POUND', 'DPOUND'] + list(
    set(reserved.values()))

literals = "+-*/%|&~^<>=!?()[]{}.,;:\\\'\""


# Whitespace
def t_WS(t):
    r'\s+'
    t.lexer.lineno += t.value.count("\n")
    return t


t_POUND = r'\#'
t_DPOUND = r'\#\#'


# Identifier
def t_ID(t):
    r'[A-Za-z_][\w_]*'
    t.type = reserved.get(t.value, 'ID')
    return t


def t_INTEGER(t):
    r'(((((0x)|(0X))[0-9a-fA-F]+)|(\d+))([uU][lL]|[lL][uU]|[uU]|[lL])?)'
    # t.value = int(t.value)
    return t


# String literal
def t_STRING(t):
    r'\"([^\\\n]|(\\(.|\n)))*?\"'
    t.lexer.lineno += t.value.count("\n")
    return t


# Character constant 'c' or L'c'
def t_CHAR(t):
    r'(L)?\'([^\\\n]|(\\(.|\n)))*?\''
    t.lexer.lineno += t.value.count("\n")
    return t


# Comment
def t_COMMENT1(t):
    r'(/\*(.|\n)*?\*/)'
    ncr = t.value.count("\n")
    t.lexer.lineno += ncr
    # replace with one space or a number of '\n'
    t.type = 'WS'
    t.value = '\n' * ncr if ncr else ' '
    return t


# Line comment
def t_COMMENT2(t):
    r'(//.*?(\n|$))'
    # replace with '/n'
    t.type = 'WS'
    t.value = '\n'
    return t


# error handling
def t_error(t):
    t.type = t.value[0]
    t.value = t.value[0]
    t.lexer.skip(1)
    print("error token encountered", t)
    return t


# rules for parsing shared memory and sync thread

shared_memory = {"symbol": [], "dtype": [], "size": []}
include_set = {}
exclude_set = {}
arguments = {"symbol": [], "dtype": []}
signature = []


def p_functon(p):
    'function : signature \'{\' shared_buffer statements \'}\''
    include_set[p.slice[2].lexpos] = p.slice[-1].lexpos
    p[0] = p.slice[-2].value


def p_signature(p):
    'signature : GLOBAL VOID ID \'(\' parameters \')\''
    signature.append(p[3])


def p_parameters(p):
    '''parameters : parameter
                  | parameters \',\' parameter
    '''


def p_parameter(p):
    '''parameter : type ID
                 | type QUALIFIER ID
                 | QUALIFIER type ID
    '''
    if (str(p.slice[1]) == "type"):
        arguments["dtype"].append(p[1])
    else:
        arguments["dtype"].append(p[2])
    arguments["symbol"].append(p[len(p) - 1])


def p_type(p):
    '''type : TYPE 
            | TYPE \'*\'
    '''
    p[0] = ''.join(p[1:])


def p_shared_buffer(p):
    '''shared_buffer : shared
                     | shared_buffer shared
                     | normal
    '''


def p_shared(p):
    'shared : SHARED TYPE ID \'[\' INTEGER \']\' \';\' '
    shared_memory["symbol"].append(p[3])
    shared_memory["dtype"].append(p[2])
    shared_memory["size"].append(int(p[5]))
    exclude_set[p.slice[1].lexpos] = p.slice[-1].lexpos


def p_statements(p):
    '''statements : statement
                  | statements statement
    '''
    p[0] = sum(p[1:])


def p_statement(p):
    '''statement : sync
                 | for_loop_static
                 | normal
    '''
    p[0] = p[1]


def p_for_loop_static(p):
    '''for_loop_static : FOR \'(\' assign compare increase \')\' \'{\' statements \'}\'
       assign   : TYPE ID \'=\' INTEGER \';\'
       compare  : ID \'<\' INTEGER \';\'
       increase : \'+\' \'+\' ID
                | ID \'+\' \'+\'
    '''
    if len(p) == 10:
        p[0] = p[4] * p[8]
    elif len(p) == 6:
        assert int(p[4]) == 0
    elif len(p) == 5:
        p[0] = int(p[3])
    else:
        pass


def p_sync(p):
    'sync : SYNC \'(\' \')\' \';\' '
    p[0] = 1


def p_normal(p):
    '''normal : \';\'
               | \'{\' normal \'}\'
    '''
    p[0] = 0


def p_error(p):
    if p:
        # Just discard the token and tell the parser it's okay.
        parser.errok()
    else:
        print("Syntax error at EOF")


lexer = lex.lex()
parser = yacc.yacc(debug=True)


def is_valid(pos):
    for start in include_set:
        # Todo: replace the static analysis part with a simpler way
        if start - 1 < pos:
            break
    else:
        return False
    for start in exclude_set:
        if start <= pos <= exclude_set[start]:
            return False
    return True


def clear_global():
    shared_memory["symbol"] = []
    shared_memory["dtype"] = []
    shared_memory["size"] = []
    include_set.clear()
    exclude_set.clear()
    arguments["symbol"] = []
    arguments["dtype"] = []


def parse(code, parameters):
    # Todo: the defined grammar for the parser is not sufficient, to be completed
    clear_global()
    num_sync = parser.parse(code)
    lexer.input(code)
    new_code = ""
    for (i, dtype) in enumerate(parameters["dtype"]):
        assert dtype == arguments["dtype"][i]
        if parameters["symbol"][i] != arguments["symbol"][i]:
            new_code += "{} {} = {};".format(dtype,
                                             arguments["symbol"][i], parameters["symbol"][i])
    while True:
        tok = lexer.token()
        if not tok:
            break
        elif is_valid(tok.lexpos):
            new_code += tok.value
        else:
            pass

    return shared_memory, num_sync, new_code, signature[-1]


if __name__ == '__main__':

    with open(sys.argv[1]) as f:
        input = f.read()

    parameters = {'symbol': ['input0', 'input1', 'output0'], 'dtype': [
        'float*', 'float*', 'float*']}
    print(parse(input, parameters)[2])
